## Training

```bash
# 2025-07-21_12-03-51_train-run1
# Reference training run used to train all three AnySleep checkpoints (anysleep-run1/2/3.pth)
python scripts/train.py -cn=exp002/exp002a
```

## Evaluation on test set

```bash
# sweep-2025-07-28_19-56-55_mass
# High-frequency predictions on mass -> arousal analysis (scripts/arousals/arousals_per_ss.ipynb) -> Fig. 2 (scripts/final-figures/fig_2.py)
python scripts/predict-high-freq.py -m -cn=exp002/exp002a +high_freq_predict.dataloader="\${data.test_dataloader}" +data.test_dataloader.dataset.datasets_to_load="['mass-c1', 'mass-c3']" model.path="anysleep-run1.pth","anysleep-run2.pth","anysleep-run3.pth" +model.sleep_stage_frequency=1,2,4,8,16,32,64,128,256,384,640,960,1920,3840

# sweep-2025-07-28_20-33-22_confusion_matrix
# Per-recording confusion matrices across EEG/EOG channel combinations -> Fig. 1a (scripts/final-figures/fig_1a.ipynb)
python scripts/predict-confusion-matrix.py -m -cn=exp002/exp002a +predict_cm.dataloader="\${data.test_dataloader}" model.path="anysleep-run1.pth","anysleep-run2.pth","anysleep-run3.pth" data.test_dataloader.dataset.n_eeg_channels=0,1,2 data.test_dataloader.dataset.n_eog_channels=0,1,2

# sweep-2025-07-29_14-59-03_test_eval
# Sleep stage metrics on the test set -> Table 1 (scripts/final-figures/table_1.ipynb),
# Table S1/S2 & Fig. S1 (scripts/final-figures/table_s1_s2-fig_s1.ipynb)
python scripts/evaluate.py -cn=exp002/exp002a -m +training.trainer.evaluators.test="\${evaluators.test}" model.path="anysleep-run1.pth","anysleep-run2.pth","anysleep-run3.pth"

# sweep-2025-07-29_18-44-18_isruc
# High-frequency predictions on the ISRUC test dataset -> Fig. S3 (scripts/final-figures/fig_s3.ipynb),
# Table S4b (scripts/final-figures/table_s4b_anysleep.json / table_s4b_anysleep_metrics.ipynb),
# downstream age/sex analysis (scripts/high-freq/anysleep_isruc_age.ipynb, anysleep_isruc_sex.ipynb)
python scripts/predict-high-freq.py -cn=exp002/exp002a -m +high_freq_predict.dataloader=$\{data.test_dataloader\} +data.test_dataloader.dataset.datasets_to_load=['isruc-sg1','isruc-sg2','isruc-sg3'] model.path="anysleep-run1.pth","anysleep-run2.pth","anysleep-run3.pth" +model.sleep_stage_frequency=1,2,4,8,16,32,64,128,256,384,640,960,1920,3840

# sweep-2025-07-30_13-56-48_test_eval_recordings
# Per-recording evaluation metrics -> Fig. S4 (scripts/final-figures/fig_s4.ipynb)
python scripts/evaluate.py -cn=exp002/exp002a -m training.trainer.evaluators=null +training.trainer.evaluators.test="\${evaluators.test}" model.path="anysleep-run1.pth","anysleep-run2.pth","anysleep-run3.pth" evaluators.test.result_tracker.sleep_stages.track_recordings=True

# sweep-2025-09-19_11-42-51_dod
# High-frequency predictions on DODH/DOOD -> Fig. 3, OSA + feature selection (scripts/final-figures/fig_3.ipynb),
# downstream analysis (scripts/high-freq/anysleep_dodo_vs_dodh.ipynb)
python scripts/predict-high-freq.py -m -cn=exp002/exp002a +high_freq_predict.dataloader="\${data.test_dataloader}" +data.test_dataloader.dataset.datasets_to_load="['dodo','dodh']" model.path="anysleep-run1.pth","anysleep-run2.pth","anysleep-run3.pth" +model.sleep_stage_frequency=1,2,4,8,16,32,64,128,256,384,640,960,1920,3840 +data.test_dataloader.dataset.channels=['C3-M2','F3-F4','F3-M2','F4-O2','F3-O1','EOG1','EOG2']

# sweep-2025-10-14_11-33-51_attention-weights
# Attention weights per channel combination on in-distribution datasets -> Fig. 4 (scripts/final-figures/fig_4.ipynb)
python scripts/evaluate.py -cn=exp002/exp002a -m training.trainer.evaluators=null +training.trainer.evaluators.test="\${evaluators.test}" +data.test_dataloader.dataset.datasets_to_load="['abc', 'chat', 'dcsm', 'hpap', 'isruc-sg1', 'isruc-sg2', 'isruc-sg3', 'phys']" +data.test_dataloader.dataset.channels="['F3-M2', 'C3-M2', 'O1-M2', 'E1-M2']" model.path="anysleep-run1.pth","anysleep-run2.pth","anysleep-run3.pth" +model.save_att_weights=True +evaluators.test.result_tracker.sleep_stages.log_channel_names=True

# sweep-2025-10-14_11-38-43_attention-weights
# Attention weights on DODO (out-of-distribution) -> Fig. 4 (scripts/final-figures/fig_4.ipynb)
python scripts/evaluate.py -cn=exp002/exp002a -m training.trainer.evaluators=null +training.trainer.evaluators.test="\${evaluators.test}" +data.test_dataloader.dataset.datasets_to_load="['dodo']" +data.test_dataloader.dataset.channels="['F3-M2', 'C3-M2', 'O1-M2', 'EOG1']" model.path="anysleep-run1.pth","anysleep-run2.pth","anysleep-run3.pth" +model.save_att_weights=True +evaluators.test.result_tracker.sleep_stages.log_channel_names=True

# sweep-2025-10-14_11-39-46_attention-weights
# Attention weights on mass-c1 -> Fig. 4 (scripts/final-figures/fig_4.ipynb)
python scripts/evaluate.py -cn=exp002/exp002a -m training.trainer.evaluators=null +training.trainer.evaluators.test="\${evaluators.test}" +data.test_dataloader.dataset.datasets_to_load="['mass-c1']" +data.test_dataloader.dataset.channels="['F3-CLE', 'C3-CLE', 'O1-CLE', 'EOG(L)']" model.path="anysleep-run1.pth","anysleep-run2.pth","anysleep-run3.pth" +model.save_att_weights=True +evaluators.test.result_tracker.sleep_stages.log_channel_names=True

# sweep-2025-10-14_11-40-44_attention-weights
# Attention weights on mass-c1 + mass-c3 -> Fig. 4 (scripts/final-figures/fig_4.ipynb)
python scripts/evaluate.py -cn=exp002/exp002a -m training.trainer.evaluators=null +training.trainer.evaluators.test="\${evaluators.test}" +data.test_dataloader.dataset.datasets_to_load="['mass-c1', 'mass-c3']" +data.test_dataloader.dataset.channels="['F3-LER', 'C3-LER', 'O1-LER', 'EOG(L)']" model.path="anysleep-run1.pth","anysleep-run2.pth","anysleep-run3.pth" +model.save_att_weights=True +evaluators.test.result_tracker.sleep_stages.log_channel_names=True

# sweep-2025-10-24_12-12-11_mass_logits
# Raw high-frequency logits on mass -> Fig. 2 (scripts/final-figures/fig_2.py)
python scripts/predict-high-freq_full_logits.py -m -cn=exp002/exp002a +high_freq_predict.dataloader="\${data.test_dataloader}" +data.test_dataloader.dataset.datasets_to_load="['mass-c1', 'mass-c3']" model.path="anysleep-run1.pth","anysleep-run2.pth","anysleep-run3.pth" +model.sleep_stage_frequency=1,2,4,8,16,32,64,128

# sweep-2025-11-24_17-53-45_nchannels
# MF1 vs. number of input channels  -> Fig. 1b (scripts/final-figures/fig_1b.ipynb)
python scripts/evaluate.py -m -cn=exp002/exp002a training.trainer.evaluators=null +training.trainer.evaluators.test="\${evaluators.test}" model.path="anysleep-run1.pth","anysleep-run2.pth","anysleep-run3.pth" +evaluators.test.result_tracker.n_channels="{_target_: base.results.anysleep_nchannels_ss_tracker.AnySleepNChannelsSSResultTracker, filename: anysleep_test_results_n_channels.json, track_datasplit: True, track_datasets: True, track_channels: False, track_recordings: True, do_majority_voting: False}" +data.test_dataloader.dataset.limit_num_samples_to=5000 data.test_dataloader.dataset.n_eeg_channels=0,1,2,3,4,5,6 data.test_dataloader.dataset.n_eog_channels=0,1 +data.test_dataloader.dataset.datasets_to_load="['abc', 'chat', 'dcsm', 'dodh', 'dodo', 'hpap', 'isruc-sg1', 'isruc-sg2', 'isruc-sg3', 'mass-c1', 'mass-c3', 'phys']"

# sweep-2026-03-02_12-24-46_test_predictions
# High-frequency predictions across all in-distribution datasets -> Table S1/S2 & Fig. S1 (scripts/final-figures/table_s1_s2-fig_s1.ipynb)
python scripts/predict-high-freq.py -m -cn=exp002/exp002a +high_freq_predict.dataloader="\${data.test_dataloader}" model.path="anysleep-run1.pth","anysleep-run2.pth","anysleep-run3.pth" +model.sleep_stage_frequency=1,2,4,8,16

# sweep-2026-03-30_17-16-31_ood_test_predictions
# High-frequency predictions on out-of-distribution test datasets -> Table S4a
# (scripts/final-figures/table_s4a_anysleep.json / table_s4a_anysleep_metrics.ipynb)
python scripts/predict-high-freq.py -m -cn=exp002/exp002a +high_freq_predict.dataloader="\${data.test_dataloader}" +data.test_dataloader.dataset.datasets_to_load=['dodh','dodo','isruc-sg1','isruc-sg2','isruc-sg3','mass-c1','mass-c3','svuh'] model.path="anysleep-run1.pth","anysleep-run2.pth","anysleep-run3.pth" +model.sleep_stage_frequency=1,2,4,8,16

# sweep-2026-04-10_19-11-06_nchannels_mass
# MF1 vs. number of input channels on mass -> Fig. S2 (scripts/final-figures/fig_s2.ipynb)
python scripts/evaluate.py -m -cn=exp002/exp002a training.trainer.evaluators=null +training.trainer.evaluators.test="\${evaluators.test}" model.path="anysleep-run1.pth","anysleep-run2.pth","anysleep-run3.pth" +evaluators.test.result_tracker.n_channels="{_target_: base.results.anysleep_nchannels_ss_tracker.AnySleepNChannelsSSResultTracker, filename: anysleep_test_results_n_channels.json, track_datasplit: True, track_datasets: True, track_channels: False, track_recordings: True, do_majority_voting: False}" +data.test_dataloader.dataset.limit_num_samples_to=5000 data.test_dataloader.dataset.n_eeg_channels=0,1,2,3,4,5,6 data.test_dataloader.dataset.n_eog_channels=0,1 +data.test_dataloader.dataset.datasets_to_load="['mass-c1','mass-c3']"

# sweep-2026-04-16_17-26-02_nchannels_mass_in-dist
# MF1 vs. number of in-distribution mass channels -> Fig. S2 (scripts/final-figures/fig_s2.ipynb)
python scripts/evaluate.py -m -cn=exp002/exp002a training.trainer.evaluators=null +training.trainer.evaluators.test="\${evaluators.test}" model.path="anysleep-run1.pth","anysleep-run2.pth","anysleep-run3.pth" +evaluators.test.result_tracker.n_channels="{_target_: base.results.anysleep_nchannels_ss_tracker.AnySleepNChannelsSSResultTracker, filename: anysleep_test_results_n_channels.json, track_datasplit: True, track_datasets: True, track_channels: False, track_recordings: True, do_majority_voting: False}" +data.test_dataloader.dataset.limit_num_samples_to=5000 data.test_dataloader.dataset.n_eeg_channels=0,1,2,3,4,5,6 data.test_dataloader.dataset.n_eog_channels=0,1 +data.test_dataloader.dataset.datasets_to_load="['mass-c1','mass-c3']" +data.test_dataloader.dataset.channels=['F3-CLE','F4-CLE','C3-CLE','C4-CLE','O1-CLE','O2-CLE','T3-CLE','T4-CLE','Fz-CLE','Cz-CLE','Pz-CLE','F3-LER','F4-LER','C3-LER','C4-LER','O1-LER','O2-LER','T3-LER','T4-LER','Fz-LER','Cz-LER','Pz-LER','Oz-LER','EOG(L)','EOG(R)']

# sweep-2026-04-17_07-06-14_nchannels_mass_ood
# MF1 vs. number of out-of-distribution mass channels -> Fig. S2 (scripts/final-figures/fig_s2.ipynb)
python scripts/evaluate.py -m -cn=exp002/exp002a training.trainer.evaluators=null +training.trainer.evaluators.test="\${evaluators.test}" model.path="anysleep-run1.pth","anysleep-run2.pth","anysleep-run3.pth" +evaluators.test.result_tracker.n_channels="{_target_: base.results.anysleep_nchannels_ss_tracker.AnySleepNChannelsSSResultTracker, filename: anysleep_test_results_n_channels.json, track_datasplit: True, track_datasets: True, track_channels: False, track_recordings: True, do_majority_voting: False}" +data.test_dataloader.dataset.limit_num_samples_to=5000 data.test_dataloader.dataset.n_eeg_channels=0,1,2,3,4,5,6 data.test_dataloader.dataset.n_eog_channels=0,1 +data.test_dataloader.dataset.datasets_to_load="['mass-c1','mass-c3']" +data.test_dataloader.dataset.channels=['F7-CLE','F8-CLE','T5-CLE','T6-CLE','P3-CLE','P4-CLE','Fp1-CLE','Fp2-CLE','F7-LER','F8-LER','T5-LER','T6-LER','P3-LER','P4-LER','Fp1-LER','Fp2-LER','A2-LER','EOG(L)','EOG(R)']
```
