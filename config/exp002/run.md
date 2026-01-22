## Training

```bash
python scripts/train.py -cn=exp002/exp002a
```

## Evaluation on test set

```bash
# Basic evaluation
python scripts/evaluate.py -cn=exp002/exp002a +training.trainer.evaluators.test="\${evaluators.test}" model.path="your_model.pth"

# Evaluation for channel matrix (2 channel combinations)
python scripts/predict-confusion-matrix.py -m -cn=exp002/exp002a +predict_cm.dataloader="\${data.test_dataloader}" model.path="your_model.pth" data.test_dataloader.dataset.n_eeg_channels=0,1,2 data.test_dataloader.dataset.n_eog_channels=0,1,2

# Evaluation for n_channels vs MF1 figure
python scripts/evaluate.py -m -cn=exp002/exp002a training.trainer.evaluators=null +training.trainer.evaluators.test="\${evaluators.test}" model.path="your_model.pth" +evaluators.test.result_tracker.n_channels="{_target_: base.results.anysleep_nchannels_ss_tracker.AnySleepNChannelsSSResultTracker, filename: anysleep_test_results_n_channels.json, track_datasplit: True, track_datasets: True, track_channels: False, track_recordings: True, do_majority_voting: False}" +data.test_dataloader.dataset.limit_num_samples_to=5000 data.test_dataloader.dataset.n_eeg_channels=0,1,2,3,4,5,6 data.test_dataloader.dataset.n_eog_channels=0,1 +data.test_dataloader.dataset.datasets_to_load="['abc', 'chat', 'dcsm', 'dodh', 'dodo', 'hpap', 'isruc-sg1', 'isruc-sg2', 'isruc-sg3', 'mass-c1', 'mass-c3', 'phys']"
```

## High-frequency prediction

```bash
# DOD datasets
python scripts/predict-high-freq.py -m -cn=exp002/exp002a +high_freq_predict.dataloader="\${data.test_dataloader}" +data.test_dataloader.dataset.datasets_to_load="['dodo', 'dodh']" model.path="your_model.pth" +model.sleep_stage_frequency=1,2,4,8,16,32,64,128,256,384,640,960,1920,3840 +data.test_dataloader.dataset.channels="['C3-M2', 'F3-F4', 'F3-M2', 'F4-O2', 'F3-O1', 'EOG1', 'EOG2']"

# ISRUC datasets
python scripts/predict-high-freq.py -m -cn=exp002/exp002a +high_freq_predict.dataloader="\${data.test_dataloader}" +data.test_dataloader.dataset.datasets_to_load="['isruc-sg1', 'isruc-sg2', 'isruc-sg3']" model.path="your_model.pth" +model.sleep_stage_frequency=1,2,4,8,16,32,64,128,256,384,640,960,1920,3840

# MASS datasets
python scripts/predict-high-freq_full_logits.py -m -cn=exp002/exp002a +high_freq_predict.dataloader="\${data.test_dataloader}" +data.test_dataloader.dataset.datasets_to_load="['mass-c1', 'mass-c3']" model.path="your_model.pth" +model.sleep_stage_frequency=1,2,4,8,16,32,64,128
```

## Channel Attention Weights

```bash
python scripts/evaluate.py -cn=exp002/exp002a training.trainer.evaluators=null +training.trainer.evaluators.test="\${evaluators.test}" +data.test_dataloader.dataset.datasets_to_load="['abc', 'chat', 'dcsm', 'hpap', 'isruc-sg1', 'isruc-sg2', 'isruc-sg3', 'phys']" +data.test_dataloader.dataset.channels="['F3-M2', 'C3-M2', 'O1-M2', 'E1-M2']" model.path="your_model.pth" +model.save_att_weights=True +evaluators.test.result_tracker.sleep_stages.log_channel_names=True
python scripts/evaluate.py -cn=exp002/exp002a training.trainer.evaluators=null +training.trainer.evaluators.test="\${evaluators.test}" +data.test_dataloader.dataset.datasets_to_load="['dodo']" +data.test_dataloader.dataset.channels="['F3-M2', 'C3-M2', 'O1-M2', 'EOG1']" model.path="your_model.pth" +model.save_att_weights=True +evaluators.test.result_tracker.sleep_stages.log_channel_names=True
python scripts/evaluate.py -cn=exp002/exp002a training.trainer.evaluators=null +training.trainer.evaluators.test="\${evaluators.test}" +data.test_dataloader.dataset.datasets_to_load="['mass-c1']" +data.test_dataloader.dataset.channels="['F3-CLE', 'C3-CLE', 'O1-CLE', 'EOG(L)']" model.path="your_model.pth" +model.save_att_weights=True +evaluators.test.result_tracker.sleep_stages.log_channel_names=True
python scripts/evaluate.py -cn=exp002/exp002a training.trainer.evaluators=null +training.trainer.evaluators.test="\${evaluators.test}" +data.test_dataloader.dataset.datasets_to_load="['mass-c1', 'mass-c3']" +data.test_dataloader.dataset.channels="['F3-LER', 'C3-LER', 'O1-LER', 'EOG(L)']" model.path="your_model.pth" +model.save_att_weights=True +evaluators.test.result_tracker.sleep_stages.log_channel_names=True
```
