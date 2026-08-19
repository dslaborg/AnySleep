## Training

```bash
# 2025-03-10_14-41-34_train-run1
# Reference training run used to train all three U-Sleep checkpoints (usleep-run1/2/3.pth)
python scripts/train.py -cn=exp001/exp001a
```

## Evaluation on test set

```bash
# sweep-2025-07-30_12-21-14_confusion_matrix
# Per-recording confusion matrices -> Fig. 1a (scripts/final-figures/fig_1a.ipynb)
python scripts/predict-confusion-matrix.py -cn=exp001/exp001a -m +predict_cm.dataloader="\${data.test_dataloader}" model.path="usleep-run1.pth","usleep-run2.pth","usleep-run3.pth" data.test_dataloader.dataset.eeg_eog_only=False

# sweep-2025-07-30_12-29-20_test_eval
# Sleep stage metrics on the test set -> Table 1 (scripts/final-figures/table_1.ipynb),
# Table S1/S2 & Fig. S1 (scripts/final-figures/table_s1_s2-fig_s1.ipynb)
python scripts/evaluate.py -cn=exp001/exp001a -m +training.trainer.evaluators.test="\${evaluators.test}" model.path="usleep-run1.pth","usleep-run2.pth","usleep-run3.pth"

# sweep-2025-07-30_14-11-25_test_eval_recordings
# Per-recording evaluation metrics -> Fig. S4 (scripts/final-figures/fig_s4.ipynb)
python scripts/evaluate.py -cn=exp001/exp001a -m +training.trainer.evaluators.test="\${evaluators.test}" model.path="usleep-run1.pth","usleep-run2.pth","usleep-run3.pth" evaluators.test.result_tracker.sleep_stages.track_recordings=True

# sweep-2025-11-24_15-26-27_nchannels
# MF1 vs. number of input channels -> Fig. 1b (scripts/final-figures/fig_1b.ipynb)
python scripts/evaluate.py -cn=exp001/exp001a -m training.trainer.evaluators=null +training.trainer.evaluators.test="\${evaluators.test}" model.path="usleep-run1.pth","usleep-run2.pth","usleep-run3.pth" +evaluators.test.result_tracker.n_channels="{_target_: base.results.usleep_nchannels_ss_tracker.USleepNChannelsSSResultTracker, filename: usleep_test_results_n_channels.json, n_channels_list: [1,2,3,4,5,6], n_samples: 5000, track_recordings: True}" +data.test_dataloader.dataset.datasets_to_load="['abc', 'chat', 'dcsm', 'dodh', 'dodo', 'hpap', 'isruc-sg1', 'isruc-sg2', 'isruc-sg3', 'mass-c1', 'mass-c3', 'phys']"

# sweep-2026-03-02_12-29-30_test_predictions
# High-frequency predictions across all in-distribution datasets -> Table S1/S2 & Fig. S1 (scripts/final-figures/table_s1_s2-fig_s1.ipynb)
python scripts/predict-high-freq.py -cn=exp001/exp001a -m +high_freq_predict.dataloader=$\{data.test_dataloader\} model.path="usleep-run1.pth","usleep-run2.pth","usleep-run3.pth" +model.sleep_stage_frequency=1,2,4,8,16

# sweep-2026-04-09_10-23-16_isruc
# High-frequency predictions on the ISRUC test datasets -> Table S4b
# (scripts/final-figures/table_s4b_usleep.json / table_s4b_usleep_metrics.ipynb)
python scripts/predict-high-freq.py -cn=exp001/exp001a -m +high_freq_predict.dataloader=$\{data.test_dataloader\} +data.test_dataloader.dataset.datasets_to_load=['isruc-sg1','isruc-sg2','isruc-sg3'] model.path="usleep-run1.pth","usleep-run2.pth","usleep-run3.pth" +model.sleep_stage_frequency=1,2,4,8,16,32,64,128,256,384,640,960,1920,3840

# sweep-2026-04-28_17-42-26_ood_test_predictions
# High-frequency predictions on out-of-distribution test datasets -> Table S4a
# (scripts/final-figures/table_s4a_usleep.json / table_s4a_usleep_metrics.ipynb)
python scripts/predict-high-freq.py -cn=exp001/exp001a -m +high_freq_predict.dataloader=$\{data.test_dataloader\} +data.test_dataloader.dataset.datasets_to_load=[dodh,dodo,isruc-sg1,isruc-sg2,isruc-sg3,mass-c1,mass-c3,svuh] model.path="usleep-run1.pth","usleep-run2.pth","usleep-run3.pth" +model.sleep_stage_frequency=1
```
