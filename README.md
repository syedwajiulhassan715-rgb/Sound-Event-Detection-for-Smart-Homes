

An end-to-end machine learning pipeline for detecting 15 domestic sound events from audio recordings, built for acoustic-based smart home monitoring. The system processes raw audio into 960-dimensional feature vectors, aggregates labels from multiple annotators using majority voting, and classifies sound events using multi-label Random Forest models.

## Overview

This project tackles the problem of recognizing overlapping household sounds (running water, footsteps, doors opening, etc.) from continuous audio recordings. It was developed as part of the Machine Learning Pattern Classification (MLPC) course and covers the full pipeline from data collection and annotation analysis through to model evaluation on a hidden test set.

The best-performing model achieves a **macro-averaged F1 score of 0.4714** on the challenge test set, a 49% improvement over the Decision Tree baseline (0.3170).

## Target Sound Classes

The system recognizes 15 domestic sound events:

`bell_ringing` · `coffee_machine` · `cutlery_dishes` · `door_open_close` · `footsteps` · `keyboard_typing` · `keychain` · `light_switch` · `microwave` · `phone_ringing` · `running_water` · `toilet_flushing` · `vacuum_cleaner` · `wardrobe_drawer_open_close` · `window_open_close`

## Project Structure

```
├── Task 1/                     # Data collection and validation
│   └── validate_submission.py  # Submission validator (duration, metadata, constraints)
├── Task 3/                     # Dataset exploration and annotation analysis
│   ├── analysis.ipynb          # Annotator agreement, class distributions, feature stats
│   ├── metadata.csv            # Recording metadata
│   └── annotations.csv         # Multi-annotator labels
├── Task 4/                     # Classification pipeline
│   └── waji.ipynb              # Label aggregation, model training, evaluation
├── Task_5_Waji/                # Challenge task
│   ├── MLPC_TASK5.ipynb        # Final pipeline with tuning and post-processing
│   └── predictions_hidden_test.csv  # Predictions on hidden test set
├── MLPC_Report_Challenge.pdf   # Project report
└── PC notes/                   # Course notes (Chapters 1-5, 7-8)
```

## Pipeline

### 1. Data Collection (Task 1)

Audio scenes were recorded in domestic environments (kitchen, bathroom, bedroom, living room, office, hallway, toilet) with both static and mobile device placements. Each recording is 15 to 35 seconds long and contains one or more overlapping target sound classes. A validation script enforces constraints including minimum scene counts, class coverage (at least 10 of 15 classes), and device placement balance (at least 3 static and 3 mobile).

### 2. Annotation Analysis (Task 3)

The dataset contains 3,656 audio files with 1 to 5 annotators per file. Analysis of inter-annotator agreement revealed:

- Overall agreement rate: **95.33%** across all class-file pairs
- Highest agreement: `toilet_flushing` (99.28%), `vacuum_cleaner` (99.14%)
- Lowest agreement: `footsteps` (84.82%), reflecting genuine ambiguity in that class
- Class imbalance: `running_water` appears in 11.3% of segments while `light_switch` appears in only 0.13%, a ratio of roughly 38:1

The most frequent co-occurrence pair is `cutlery_dishes` + `running_water` with 1,904 overlapping segments.

### 3. Feature Extraction

Each 1-second audio segment (with 0.5-second hop) is represented by a **960-dimensional feature vector** computed from:

- Zero Crossing Rate (ZCR)
- Mel spectrogram bands
- MFCCs with first and second order deltas
- Spectral flux, flatness, centroid, bandwidth, contrast, and rolloff
- Short-term energy and power

For each feature, four statistics are computed: mean, standard deviation, minimum, and maximum.

### 4. Label Aggregation

Multi-annotator labels are aggregated using **majority voting**: a sound class is considered present in a segment only if more than half the annotators marked it as active. This produces binary multi-label ground truth for each segment.

### 5. Classification (Tasks 4 and 5)

**Data splitting** uses collector-level `GroupShuffleSplit` to prevent data leakage. Files from the same collector never appear in both training and evaluation sets. The split yields 285 / 61 / 62 collectors for train / validation / test.

**Preprocessing**: `StandardScaler` fitted on training data only, then applied to validation and test sets.

**Models evaluated**:

| Model | Configuration | Macro F1 |
|-------|--------------|----------|
| Always-zero baseline | Predicts no active classes | 0.0000 |
| Random baseline | Uniform random predictions | 0.0517 |
| Decision Tree | Default, balanced weights | 0.3170 |
| Random Forest (Task 4) | 100 trees, default depth | 0.3316 |
| **Random Forest (Task 5)** | **200 trees, depth=20, leaf=5, balanced** | **0.4714** |
| KNN | k=5, subsampled to 20k segments | 0.2804 |

All classifiers are wrapped in sklearn's `MultiOutputClassifier` for multi-label support.

### 6. Post-Processing

Temporal **median filtering** was explored with window sizes of 1, 3, 5, 7, and 9 frames to smooth predictions across consecutive segments. The effect varies by class and window size.

## Key Results

- Best model: **Random Forest** with `n_estimators=200`, `max_depth=20`, `min_samples_leaf=5`, `class_weight='balanced'`
- Challenge test set macro F1: **0.4714**
- Improvement over Decision Tree baseline: **+48.7%**
- Per-class performance varies significantly due to class imbalance and acoustic overlap between classes

## Requirements

- Python 3.x
- scikit-learn
- pandas
- numpy
- librosa
- matplotlib
- seaborn
- pydub

## Usage

The main classification pipeline is in the Jupyter notebooks. To reproduce results:

1. Place the dataset files (audio features and annotations) in the appropriate task directories
2. Run `Task 3/analysis.ipynb` for dataset exploration
3. Run `Task 4/waji.ipynb` for the initial classification experiments
4. Run `Task_5_Waji/MLPC_TASK5.ipynb` for the final tuned pipeline and challenge predictions

To validate a data collection submission:

```bash
python "Task 1/validate_submission.py" submission.zip
```

## Author

Syed Wajiul Hassan
Rayan Ali Javed

---

