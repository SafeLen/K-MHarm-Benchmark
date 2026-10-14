"""
모델 비교 실험용 LoRA 학습 (기본 코드). --model 하나만 바꿔서 다섯 모델을 똑같이 학습한다.

    python train_lora.py --model varco        --output-dir outputs/varco
    python train_lora.py --model hyperclovax  --output-dir outputs/hyperclovax
    python train_lora.py --model bllossom     --output-dir outputs/bllossom
    python train_lora.py --model ax           --output-dir outputs/ax
    python train_lora.py --model qwen3vl      --output-dir outputs/qwen3vl   # transformers>=4.57

모든 모델에 똑같이 적용되는 설정 (run_info.json에 기록됨)
    - 데이터: data_prep.py가 만든 train.jsonl / val.jsonl (유유/유무/무유/무무 모두 포함)
    - batch size 1 + grad_accum 16, epochs 3, lr 2e-4, cosine 스케줄, seed 42
    - LoRA r=16, alpha=32, dropout=0.05, 언어모델 q/k/v/o/gate/up/down_proj에만 적용
    - loss: 정답 토큰에 대한 plain cross-entropy
    - early stopping: val loss 2 epoch 연속 미개선 시 종료, val loss가 가장 낮았던 가중치로 복원
    - 임계값: val에서 F1(hate) 최대 지점을 한 번 고르고 test에는 그 값을 고정 적용

저장 파일 (--output-dir)
    adapter_model.safetensors 등   LoRA 어댑터
    run_info.json / run_info.csv   설정, 파라미터 수, best epoch, 임계값, 학습 시간, GPU
    training_log.csv               optimizer step마다 train loss, lr, grad norm
    epoch_log.csv                  epoch마다 train loss 평균, val loss, 소요 시간
    threshold_search.csv           val에서 임계값 후보별 Acc/P/R/F1/Specificity
    val_predictions.csv            val 표본별 점수/예측

기존 VARCO 전용 튜닝(focal loss, label smoothing, stage 2 케이스 추가학습, case-balance,
hard-example 가중치, aux modality loss, modality dropout, precision_constrained 임계값)은
모델 비교의 공정성을 위해 모두 뺐다.
"""
import argparse
import datetime
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch

import image_prep
from common import is_hate
from models import count_params, get_adapter, hate_score, target_logprob


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path, limit=None):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    return recs[:limit] if limit else recs


def filter_readable(records, dataset_root):
    good = []
    for r in records:
        if image_prep.load_image_for_model(os.path.join(dataset_root, r["image_path"]), sample_id=r["id"]) is not None:
            good.append(r)
    return good


def example_loss(adapter, model, record, dataset_root, split):
    """한 표본의 평균 NLL (정답 토큰 기준)."""
    image = image_prep.prepare_image(os.path.join(dataset_root, record["image_path"]), split, sample_id=record["id"])
    base = adapter.prompt_inputs(image, record["prompt"])
    inputs, tid = adapter.append_target(base, record["target"])
    return -target_logprob(model, inputs, tid) / tid.shape[0]


@torch.no_grad()
def val_loss(adapter, model, val_records, dataset_root):
    model.eval()
    losses = [example_loss(adapter, model, r, dataset_root, "val").item() for r in val_records]
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def select_threshold_on_val(adapter, model, val_records, dataset_root):
    """val에서 hate-score 임계값을 F1(hate) 최대 기준으로 고른다.
    (버그 수정) 정답을 label=="hate"가 아니라 target=="혐오"로 판정 -> 유무/무유도 유해(1)로 처리."""
    model.eval()
    scores, labels = [], []
    for r in val_records:
        image = image_prep.prepare_image(os.path.join(dataset_root, r["image_path"]), "val", sample_id=r["id"])
        s, _, _ = hate_score(adapter, model, image, r["prompt"])
        scores.append(s)
        labels.append(is_hate(r))
    model.train()

    val_pred = [{"id": r["id"], "label": r["label"], "y_true": y, "score": s}
                for r, y, s in zip(val_records, labels, scores)]
    if len(set(labels)) < 2:
        print("[임계값] val이 단일 클래스라 기본값 0.0 사용")
        return 0.0, None, [], val_pred

    # 후보: 정렬된 점수의 인접 중간값 (경계에 걸린 표본이 애매하지 않도록)
    s_sorted = sorted(set(scores))
    candidates = [s_sorted[0] - 1e-6] + [(a + b) / 2 for a, b in zip(s_sorted[:-1], s_sorted[1:])]
    y = np.array(labels)
    sc = np.array(scores)
    best_t, best_f1, search = 0.0, -1.0, []
    for t in candidates:
        p = (sc > t).astype(int)
        tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        search.append({"threshold": float(t), "f1_hate": f1, "accuracy": (tp + tn) / len(y),
                       "precision": tp / (tp + fp) if (tp + fp) else 0.0, "recall": tp / (tp + fn) if (tp + fn) else 0.0,
                       "specificity": tn / (tn + fp) if (tn + fp) else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    for v in val_pred:
        v["y_pred"] = int(v["score"] > best_t)
    print(f"[임계값] val n={len(scores)} -> threshold={best_t:.4f}, val F1(hate)={best_f1:.3f}")
    return best_t, best_f1, search, val_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["varco", "hyperclovax", "bllossom", "ax", "qwen3vl"])
    ap.add_argument("--dataset-root", default="../image_dataset")
    ap.add_argument("--train-jsonl", default="data/train.jsonl")
    ap.add_argument("--val-jsonl", default="data/val.jsonl")
    ap.add_argument("--output-dir", default=None, help="기본값: outputs/<model>")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--early-stopping-patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="스모크 테스트용: train/val을 앞에서 N개만 사용 (본 실험에서는 쓰지 말 것)")
    args = ap.parse_args()

    out_dir = args.output_dir or os.path.join("outputs", args.model)
    os.makedirs(out_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    adapter = get_adapter(args.model, device)
    model = adapter.load()
    base_total_params, _ = count_params(model)
    print(f"[{adapter.display}] base 파라미터 수: {base_total_params / 1e9:.2f}B")

    from peft import LoraConfig, get_peft_model
    targets = adapter.lora_target_modules(model)
    print(f"LoRA 적용 층 수: {len(targets)} (예: {targets[0]})")
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    _, lora_params = count_params(model)
    # peft가 자동으로 켜는 input-require-grads 훅 제거 (모든 모델 동일 조건, HyperCLOVAX in-place 에러 방지)
    try:
        model.get_base_model().disable_input_require_grads()
        print("input-require-grads 훅 제거")
    except Exception:
        pass

    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    except Exception as e:  # 일부 remote-code 모델은 미지원 -> 메모리가 충분하면 그대로 진행
        print(f"[경고] gradient checkpointing 사용 불가: {e}")
    model.train()

    train_records = filter_readable(load_jsonl(args.train_jsonl, args.limit), args.dataset_root)
    val_records = filter_readable(load_jsonl(args.val_jsonl, args.limit), args.dataset_root)
    print(f"train {len(train_records)} / val {len(val_records)}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    steps_per_epoch = math.ceil(len(train_records) / args.grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps_per_epoch * args.epochs, 1))

    order_rng = random.Random(args.seed)
    best_val, best_state, no_improve, epochs_run, best_epoch = float("inf"), None, 0, 0, None
    step_log, epoch_log = [], []   # -> training_log.csv (optimizer step 단위), epoch_log.csv
    global_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        t_ep = time.time()
        order = list(range(len(train_records)))
        order_rng.shuffle(order)
        optimizer.zero_grad()
        accum_losses, epoch_losses = [], []
        for i, idx in enumerate(order):
            loss = example_loss(adapter, model, train_records[idx], args.dataset_root, "train")
            (loss / args.grad_accum).backward()
            accum_losses.append(loss.item()); epoch_losses.append(loss.item())
            last = (i + 1 == len(order))
            if (i + 1) % args.grad_accum == 0 or last:  # 마지막 남은 누적분도 버리지 않고 반영
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                lr_now = scheduler.get_last_lr()[0]
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                global_step += 1
                step_log.append({"epoch": epoch, "step": global_step, "samples_seen": epoch * len(order) + i + 1,
                                 "train_loss": float(np.mean(accum_losses)), "lr": lr_now, "grad_norm": grad_norm})
                accum_losses = []
                if global_step % 10 == 0:
                    print(f"epoch {epoch} step {global_step} [{i + 1}/{len(order)}] "
                          f"train loss {np.mean([s['train_loss'] for s in step_log[-10:]]):.4f}")
        epochs_run = epoch + 1

        vl = val_loss(adapter, model, val_records, args.dataset_root)
        epoch_log.append({"epoch": epoch, "train_loss_mean": float(np.mean(epoch_losses)), "val_loss": vl,
                          "minutes": round((time.time() - t_ep) / 60, 2)})
        pd.DataFrame(step_log).to_csv(os.path.join(out_dir, "training_log.csv"), index=False)
        pd.DataFrame(epoch_log).to_csv(os.path.join(out_dir, "epoch_log.csv"), index=False)
        print(f"=== epoch {epoch} train loss {np.mean(epoch_losses):.4f} | val loss {vl:.4f} ===")
        if vl < best_val:
            best_epoch = epoch
            best_val, no_improve = vl, 0
            # (버그 수정) best 가중치를 메모리에 복사해두고 마지막에 확실히 복원한다.
            # 기존 코드의 load_state_dict(strict=False)는 peft 키 이름이 달라 아무것도 복원되지 않을 수 있었음.
            best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        else:
            no_improve += 1
            if args.early_stopping_patience and no_improve >= args.early_stopping_patience:
                print(f"early stopping (best val loss {best_val:.4f})")
                break

    if best_state is not None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in best_state:
                    p.copy_(best_state[n].to(p.device, p.dtype))
        print(f"best val loss({best_val:.4f}) 시점의 가중치로 복원")
    train_minutes = (time.time() - t0) / 60

    model.save_pretrained(out_dir)
    threshold, val_f1, search, val_pred = select_threshold_on_val(adapter, model, val_records, args.dataset_root)
    pd.DataFrame(search).to_csv(os.path.join(out_dir, "threshold_search.csv"), index=False)
    pd.DataFrame(val_pred).to_csv(os.path.join(out_dir, "val_predictions.csv"), index=False, encoding="utf-8-sig")

    run_info = {
        "model_key": args.model, "model_id": adapter.hf_id, "model_display": adapter.display,
        "base_params": base_total_params, "lora_trainable_params": lora_params,
        "n_lora_target_modules": len(targets),
        "train_size": len(train_records), "val_size": len(val_records),
        "epochs": args.epochs, "epochs_run": epochs_run, "batch_size": 1, "grad_accum": args.grad_accum,
        "lr": args.lr, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
        "early_stopping_patience": args.early_stopping_patience, "seed": args.seed,
        "image_size": image_prep.IMAGE_SIZE, "best_val_loss": best_val, "best_epoch": best_epoch,
        "optimizer_steps": global_step, "lora_target_module_example": targets[0],
        "selected_threshold": threshold, "val_f1_hate_at_threshold": val_f1,
        "train_minutes": round(train_minutes, 1),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "limit": args.limit, "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    pd.DataFrame([run_info]).to_csv(os.path.join(out_dir, "run_info.csv"), index=False, encoding="utf-8-sig")
    print(f"저장 완료: {out_dir} (adapter, run_info.json/csv, training_log.csv, epoch_log.csv, "
          f"threshold_search.csv, val_predictions.csv)")


if __name__ == "__main__":
    main()
