## Training

```bash
python scripts/train.py -cn=exp001/exp001a
```

## Evaluation on early stopping set

```bash
python scripts/evaluate.py -cn=exp001/exp001a data.earlystopping_dataloader.dataset.all_channels=True model.path="your_model.pth"
```

## Evaluation on test set

```bash
# base eval
python scripts/evaluate.py -cn=exp001/exp001a +training.trainer.evaluators.test="\${evaluators.test}" model.path="your_model.pth"

# evaluation for channel matrix of 2 channel combinations
python scripts/predict-confusion-matrix.py -cn=exp001/exp001a +predict_cm.dataloader="\${data.test_dataloader}" model.path="your_model.pth" data.test_dataloader.dataset.eeg_eog_only=False

# evaluation for figure n_channels vs MF1
python scripts/evaluate.py -cn=exp001/exp001a training.trainer.evaluators=null +training.trainer.evaluators.test="\${evaluators.test}" model.path="your_model.pth" +evaluators.test.result_tracker.n_channels="{_target_: base.results.usleep_nchannels_ss_tracker.USleepNChannelsSSResultTracker, filename: usleep_test_results_n_channels.json, n_channels_list: [1,2,3,4,5,6], n_samples: 5000, track_recordings: True}" +data.test_dataloader.dataset.datasets_to_load="['abc', 'chat', 'dcsm', 'dodh', 'dodo', 'hpap', 'isruc-sg1', 'isruc-sg2', 'isruc-sg3', 'mass-c1', 'mass-c3', 'phys']"
```
