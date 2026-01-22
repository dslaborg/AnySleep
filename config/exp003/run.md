## Training

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py -cn=exp003/exp003a
```

## Evaluation on test set

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py -cn=exp004/exp004a +training.trainer.evaluators.test="\${evaluators.test}" model.path="your_model.pth"
```
