## Training

```bash
# 2025-07-28_20-28-42_train-run1
# Reference training run used to train all three AnySleepEF (early fusion) checkpoints
CUDA_VISIBLE_DEVICES=0 python scripts/train.py -cn=exp003/exp003a
```

## Evaluation on test set

```bash
# sweep-2025-08-04_10-57-43_test_eval
# Sleep stage metrics on the test set -> Table 1 (scripts/final-figures/table_1.ipynb)
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py -cn=exp003/exp003a -m +training.trainer.evaluators.test="\${evaluators.test}" model.path="anysleep-early-fusion-run1.pth","anysleep-early-fusion-run2.pth","anysleep-early-fusion-run3.pth"
```
