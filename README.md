# CS4486 Mini-Project — Mechanism of Action Prediction (lish-moa)

Group 3 · Kaggle competition: [lish-moa](https://www.kaggle.com/competitions/lish-moa)

| Member | Student ID |
|---|---|
| Zhang Kaixuan (leader) | 72400703 |
| Li Tianming | 72401183 |
| Cui Hongrui | 72400021 |

## Results

| Model | 5-fold OOF log loss |
|---|---|
| Baseline MLP | 0.02023 |
| + label smoothing 0.001 | 0.01932 |
| + seed averaging (42, 123) | **0.01886** |
| Wider MLP (2048) | 0.02472 |

Kaggle private score: **0.01852** (public 0.02045). The fifth-place private score was 0.01602, so we are 0.0025 behind the top-5 level.

## Method

- Drop the 1,866 control samples, keep 21,948 treated rows and 3,982 test rows.
- 874 inputs: 772 gene-expression + 100 cell-viability + cp_dose + cp_time.
- MLP 874-1024-1024-206, batch norm, dropout 0.5, BCEWithLogits, Adam 1e-3.
- 35 epochs, batch size 128, 5-fold cross-validation with a fixed fold seed.
- Label smoothing on the targets, then probability averaging over two model seeds on identical folds.

## Repository layout

```
train.py                  training script for all four configurations
kernel/main.py            Kaggle notebook version (submitted to the leaderboard)
kernel/kernel-metadata.json
results/results*.json     OOF log loss for each configuration
results/oof*.npy          out-of-fold predictions (21948 x 206)
results/test_pred*.npy    test predictions (3982 x 206)
results/targets.json      column order for the 206 targets
submission.csv            final Kaggle submission
Group3_Project2.pdf       presentation slides
```

## Reproduce

```bash
pip install torch pandas numpy scikit-learn

export MOA_DATA=/path/to/lish-moa      # folder with train_features.csv, train_targets_scored.csv, test_features.csv
python train.py --out results --tag ""                      # baseline
python train.py --out results --tag _smooth --smooth 0.001  # + label smoothing
python train.py --out results --tag _seed123 --model_seed 123  # second seed
python train.py --out results --tag _wide --hidden 2048     # wider network
```

Each run prints the fold losses and writes `results<tag>.json`, `oof<tag>.npy` and `test_pred<tag>.npy`. The final OOF is the element-wise mean of `oof_smooth.npy` and `oof_seed123.npy`. A full run takes about 20 minutes on an RTX 4060.

## Validation protocol

- 5 folds, shuffled with fold seed 42, identical splits for every configuration.
- Loss and metric are both mean column-wise log loss, the official competition metric.
- No test labels, no external data, no pretrained model.
- The folds are plain shuffled KFold, not multilabel-stratified; switching to `MultilabelStratifiedKFold` is listed as future work.

## Kaggle submission

lish-moa is a code competition, so the CSV upload tab is disabled. `kernel/main.py` is pushed as a Kaggle notebook with the competition data attached, and the notebook output is the submitted file.

## AI usage

An AI coding assistant (Trae IDE) was used for code scaffolding, debugging, API lookup and figure generation. The group chose the competition and protocol, reviewed the pipeline, and re-ran every reported configuration. No public notebook or solution was copied.
