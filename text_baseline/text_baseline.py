"""
텍스트 전용 분류기 baseline (작업 3, RQ1 근거).

KcELECTRA를 UnSmile train으로 학습하고, test 600건의 텍스트만 넣어 게시물 단위 정답으로 평가한다.
이미지를 전혀 보지 않으므로 조건 10(유해 이미지 + 무해 텍스트)은 놓쳐야 정상이다.
조건 10의 텍스트는 비혐오 게시물(00)의 텍스트를 그대로 쓰므로, 이 모델의 Det@10은
FPR@00과 같아진다. 즉 텍스트만 보는 평가로는 이미지에만 있는 유해성을 잴 수 없다.

라벨 재정의 (논문 3.1과 동일)
    혐오  : 8개 혐오 범주 중 하나 이상이 1
    무해  : clean=1 이고 혐오 범주·악플/욕설이 모두 0
    제외  : 악플/욕설만 있는 문장
누수 방지: test 600건의 텍스트(고유 300개)와 같은 문장은 UnSmile train/valid에서 뺀다.
epoch 선택과 임계값: UnSmile valid (같은 재정의, test 문장 제외)에서 AUROC로 epoch를 고르고,
    eval_common.py가 valid 점수로 BA 최대 임계값을 고른다. test로는 아무것도 고르지 않는다.

출력: preds/text_kcelectra_seed{seed}.csv (공통 형식, input_mode=TO)
    split=val 행은 UnSmile valid이며 condition을 혐오 01 / 무해 00으로 적는다 (임계값 선택용).

Usage
    python text_baseline.py --seed 1        # 2, 3도 반복
    python ../evaluation/eval_common.py --pred "../preds/text_kcelectra_seed*.csv" --out ../results/text_kcelectra
"""
import argparse
import json
import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HATE = ["여성/가족", "남성", "성소수자", "인종/국적", "연령", "지역", "종교", "기타 혐오"]
COND = {"hate": "11", "image_hate_text_safe": "10", "text_hate_image_safe": "01", "non_hate": "00"}


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def load_unsmile(path, exclude):
    df = pd.read_csv(path, sep="\t")
    df["text"] = df["문장"].map(norm)
    hate = df[HATE].sum(axis=1) > 0
    clean = (df["clean"] == 1) & (df[HATE + ["악플/욕설"]].sum(axis=1) == 0)
    df = df[hate | clean].assign(y=hate[hate | clean].astype(int))
    df = df[~df["text"].isin(exclude)].drop_duplicates("text")
    return df[["text", "y"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unsmile-dir", default=os.path.join(HERE, "..", "data_construction", "unsmile"))
    # 논문 결과는 main_experiment의 seed1 test.jsonl로 냈다. test의 고유 텍스트 300개와 조건별 텍스트 구성
    # (조건 10 = 무해 텍스트 150개 전부, 조건 11·01 = 혐오 텍스트 150개 전부)은 어느 test.jsonl이든 같아서 지표도 같다.
    ap.add_argument("--test-jsonl", default=os.path.join(HERE, "..", "vlm_compare", "data", "test.jsonl"))
    ap.add_argument("--model", default="beomi/KcELECTRA-base-v2022")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="빠른 점검용: train을 이 개수로 자름")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "preds"))
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    with open(args.test_jsonl, encoding="utf-8") as f:
        test = [json.loads(l) for l in f]
    test_texts = {norm(r["source_text_raw"]) for r in test}
    train = load_unsmile(os.path.join(args.unsmile_dir, "unsmile_train_v1.0.tsv"), test_texts)
    valid = load_unsmile(os.path.join(args.unsmile_dir, "unsmile_valid_v1.0.tsv"), test_texts | set(train["text"]))
    if args.limit:
        train = train.sample(n=min(args.limit, len(train)), random_state=args.seed)
        valid = valid.sample(n=min(args.limit // 4 or 1, len(valid)), random_state=args.seed)
    print(f"train {len(train)} (혐오 {train.y.sum()}), valid {len(valid)} (혐오 {valid.y.sum()}), "
          f"test {len(test)} (고유 텍스트 {len(test_texts)}), device {device}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, 1 - s / steps))

    def scores(texts):
        model.eval()
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), 128):
                enc = tok(texts[i:i + 128], truncation=True, max_length=args.max_len,
                          padding=True, return_tensors="pt").to(device)
                logits = model(**enc).logits.float()
                out += (logits[:, 1] - logits[:, 0]).tolist()
        return np.array(out)

    uniq = sorted(test_texts)
    best_auc, best = -1.0, None
    texts, ys = train["text"].tolist(), train["y"].tolist()
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(texts)))
        random.shuffle(order)
        for b, i in enumerate(range(0, len(order), args.bs)):
            idx = order[i:i + args.bs]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(device)
            loss = model(**enc, labels=torch.tensor([ys[j] for j in idx], device=device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            if b % 50 == 0:
                print(f"  epoch {epoch + 1} batch {b}/{len(order) // args.bs} loss {loss.item():.4f} "
                      f"({(time.time() - t0) / 60:.1f}분)", flush=True)
        v = scores(valid["text"].tolist())
        auc = roc_auc_score(valid["y"], v)
        print(f"epoch {epoch + 1} valid AUROC {auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc, best = auc, {"valid": v, "test": dict(zip(uniq, scores(uniq))), "epoch": epoch + 1}

    rows = []
    name, setting = "KcELECTRA-base (text-only)", "trained"
    for t, y, s in zip(valid["text"], valid["y"], best["valid"]):
        rows.append({"sample_id": f"unsmile_valid_{len(rows)}", "condition": "01" if y else "00",
                     "input_mode": "TO", "score": s, "split": "val"})
    for r in test:
        rows.append({"sample_id": r["id"], "condition": COND[r["label"]], "input_mode": "TO",
                     "score": best["test"][norm(r["source_text_raw"])], "split": "test"})
    df = pd.DataFrame(rows).assign(model=name, setting=setting, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"text_kcelectra_seed{args.seed}.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    info = {"model": args.model, "seed": args.seed, "best_epoch": best["epoch"], "valid_auroc": best_auc,
            "n_train": len(train), "n_valid": len(valid), "lr": args.lr, "bs": args.bs, "epochs": args.epochs,
            "max_len": args.max_len, "minutes": round((time.time() - t0) / 60, 1), "device": device}
    with open(path.replace(".csv", "_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(path, info)


if __name__ == "__main__":
    main()
