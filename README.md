---
dataset_info:
  features:
  - name: comment_text
    dtype: string
  - name: non_toxic
    dtype: int64
  - name: toxic
    dtype: float64
  - name: severe_toxic
    dtype: float64
  - name: obscene
    dtype: float64
  - name: threat
    dtype: float64
  - name: insult
    dtype: float64
  - name: identity_hate
    dtype: float64
  splits:
  - name: train
    num_bytes: 816562684
    num_examples: 2223065
  download_size: 478320245
  dataset_size: 816562684
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
