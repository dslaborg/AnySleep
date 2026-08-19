# High-frequency sleep stage analyses

Notebooks that analyze whether high-frequency (1-3840 s sleep stage resolution) predictions from the
`predict-high-freq.py` sweeps capture disorder-, age-, or sex-related information beyond the standard 30 s resolution.
All classifiers/regressors are random forests trained on sleep stage transition patterns (triplet features) at different
time scales.

## Disorders

Three files per disorder (NT1, OSA, insomnia):

- `anysleep_<disorder>.ipynb` creates the scores for figure 3 in the manuscript (sleep stage resolution vs. MF1 score)
- `anysleep_<disorder>_sfs.ipynb` performs a sequential feature selection (sfs) run on the best performing sampling rate
  from `anysleep_<disorder>.ipynb`; the results are saved in `logs/anysleep_<disorder>_sfs.log`
- `analyze_<disorder>_sfs.ipynb` parses `logs/anysleep_<disorder>_sfs.log` and extracts the most often selected features

| Disorder           | Scores (fig. 3)               | SFS run                           | SFS analysis                | SFS log                              |
|--------------------|-------------------------------|-----------------------------------|-----------------------------|--------------------------------------|
| NT1 (MNC datasets) | `anysleep_mnc_nt1.ipynb`      | `anysleep_mnc_nt1_sfs.ipynb`      | `analyze_mnc_nt1_sfs.ipynb` | `logs/anysleep_mnc_nt1_sfs.log`      |
| OSA (DODO vs DODH) | `anysleep_dodo_vs_dodh.ipynb` | `anysleep_dodo_vs_dodh_sfs.ipynb` | `analyze_dod_sfs.ipynb`     | `logs/anysleep_dodo_vs_dodh_sfs.log` |
| Insomnia (CAP)     | `anysleep_cap_insomnia.ipynb` | `anysleep_cap_insomnia_sfs.ipynb` | `analyze_cap_ins_sfs.ipynb` | `logs/anysleep_cap_insomnia_sfs.log` |

## Age and sex

`anysleep_isruc_age.ipynb` and `anysleep_isruc_sex.ipynb` only contain the scores for figure s3 in the appendix of the
manuscript (sleep stage resolution vs. MF1 score).

## Structure

```
high-freq/
├── logs/                       # sequential feature selection logs
│   ├── anysleep_cap_insomnia_sfs.log       # insomnia sfs results
│   ├── anysleep_dodo_vs_dodh_sfs.log       # OSA (DODO vs DODH) sfs results
│   └── anysleep_mnc_nt1_sfs.log            # NT1 sfs results
├── analyze_cap_ins_sfs.ipynb   # parse insomnia sfs log, most often selected features
├── analyze_dod_sfs.ipynb       # parse OSA sfs log, most often selected features
├── analyze_mnc_nt1_sfs.ipynb   # parse NT1 sfs log, most often selected features
├── anysleep_cap_insomnia.ipynb         # insomnia resolution vs. MF1 (fig. 3)
├── anysleep_cap_insomnia_sfs.ipynb     # insomnia sfs on best sampling rate
├── anysleep_dodo_vs_dodh.ipynb         # OSA resolution vs. MF1 (fig. 3)
├── anysleep_dodo_vs_dodh_sfs.ipynb     # OSA sfs on best sampling rate
├── anysleep_isruc_age.ipynb            # age resolution vs. MF1 (fig. s3)
├── anysleep_isruc_sex.ipynb            # sex resolution vs. MF1 (fig. s3)
├── anysleep_mnc_nt1.ipynb              # NT1 resolution vs. MF1 (fig. 3)
├── anysleep_mnc_nt1_sfs.ipynb          # NT1 sfs on best sampling rate
└── rnd_parameter_map_rndfcls.json      # random parameters for the 50 random forest trainings in each anysleep_* script
```
