## Training

```bash
# 2025-07-28_20-29-58_train-run1
# Reference training run used to train all three AnySleepLF (late fusion) checkpoints
CUDA_VISIBLE_DEVICES=0 python scripts/train.py -cn=exp004/exp004a
```

## Evaluation on test set

```bash
# sweep-2025-08-04_10-58-11_test_eval
# Sleep stage metrics on the test set -> Table 1 (scripts/final-figures/table_1.ipynb)
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py -cn=exp004/exp004a -m +training.trainer.evaluators.test="\${evaluators.test}" model.path="anysleep-late-fusion-run1.pth","anysleep-late-fusion-run2.pth","anysleep-late-fusion-run3.pth"
```
