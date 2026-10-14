| Model | Setting | Input | seeds | Acc | BA | AUROC | Macro-F1 | FPR@00 | Det@11 | Det@10 | Det@01 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A.X-4.0-VL-Light-8B | fine-tuned (comparison) | IO | 1 | 72.7 | 79.3 | 82.4 | 70.6 | 7.3 | 95.3 | 95.3 | 7.3 |
| A.X-4.0-VL-Light-8B | fine-tuned (comparison) | MM | 1 | 95.0 | 94.9 | 99.0 | 93.5 | 5.3 | 95.3 | 98.0 | 92.0 |
| A.X-4.0-VL-Light-8B | fine-tuned (comparison) | TO | 1 | 64.3 | 74.7 | 80.2 | 63.3 | 4.7 | 78.7 | 4.7 | 78.7 |
| Bllossom-AICA-5B | fine-tuned (comparison) | IO | 1 | 71.3 | 79.8 | 82.2 | 69.7 | 3.3 | 92.7 | 92.7 | 3.3 |
| Bllossom-AICA-5B | fine-tuned (comparison) | MM | 1 | 91.7 | 90.0 | 96.8 | 89.1 | 13.3 | 98.0 | 94.0 | 88.0 |
| Bllossom-AICA-5B | fine-tuned (comparison) | TO | 1 | 71.0 | 71.1 | 78.2 | 66.9 | 28.7 | 92.0 | 28.7 | 92.0 |
| HyperCLOVAX-SEED-Vision-3B | fine-tuned (comparison) | IO | 1 | 70.3 | 79.1 | 82.7 | 68.8 | 3.3 | 90.7 | 90.7 | 3.3 |
| HyperCLOVAX-SEED-Vision-3B | fine-tuned (comparison) | MM | 1 | 96.3 | 94.4 | 99.0 | 95.0 | 9.3 | 98.0 | 98.0 | 98.7 |
| HyperCLOVAX-SEED-Vision-3B | fine-tuned (comparison) | TO | 1 | 72.8 | 68.8 | 79.6 | 66.8 | 39.3 | 96.0 | 39.3 | 95.3 |
| Qwen3-VL-8B-Instruct | fine-tuned (comparison) | IO | 1 | 71.7 | 80.2 | 82.8 | 70.1 | 2.7 | 93.3 | 93.3 | 2.7 |
| Qwen3-VL-8B-Instruct | fine-tuned (comparison) | MM | 1 | 93.7 | 93.1 | 98.1 | 91.8 | 8.0 | 97.3 | 96.7 | 88.7 |
| Qwen3-VL-8B-Instruct | fine-tuned (comparison) | TO | 1 | 66.7 | 72.9 | 78.7 | 64.6 | 14.7 | 83.3 | 14.7 | 83.3 |
| VARCO-VISION-2.0-1.7B | fine-tuned (comparison) | IO | 1 | 71.3 | 78.4 | 82.2 | 69.4 | 7.3 | 92.7 | 92.7 | 7.3 |
| VARCO-VISION-2.0-1.7B | fine-tuned (comparison) | MM | 1 | 91.3 | 90.7 | 97.6 | 88.9 | 10.7 | 94.0 | 96.0 | 86.0 |
| VARCO-VISION-2.0-1.7B | fine-tuned (comparison) | TO | 1 | 46.0 | 63.8 | 76.9 | 45.9 | 0.7 | 42.0 | 0.7 | 42.0 |

값은 %, seed가 2개 이상이면 평균±표준편차. 정답은 게시물 단위(조건 11·10·01=혐오, 00=비혐오).
Det@c는 조건 c의 혐오 판정 비율, FPR@00은 비혐오 게시물의 혐오 판정 비율.
