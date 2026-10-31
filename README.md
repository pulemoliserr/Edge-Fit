# EdgeFit — On-Device Human Activity Recognition

**Can a wearable recognise what its user is doing — on the device, for people it has never seen?**

This project compares six machine-learning candidates on real smartphone sensor data from 30 people and
recommends a model for on-device deployment, weighing accuracy against model size, latency, feature burden
and interpretability. An interactive Streamlit app presents the results, a business-case calculator and a
live demo on real sensor windows from unseen users.

🔗 **Live app:** `https://<your-app>.streamlit.app` · 📊 **Data:** [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

---

## Key results (unseen users 25–30)

| Model | Features | Macro F1 | Accuracy | Size | CPU latency / window (ms) | Train time (s) |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | 561 | **0.966** | **96.4%** | **25 KB** | 0.29 | 1.5 |
| LightGBM | 561 | 0.943 | 94.1% | 1,778 KB | 0.95 | 77.7 |
| Random Forest | 561 | 0.927 | 92.8% | 4,229 KB | 56.09 | 17.9 |
| Logistic Regression (top 25) | 25 | 0.897 | 89.9% | 2 KB | 0.29 | 0.2 |
| LightGBM (top 25) | 25 | 0.883 | 88.1% | 2,320 KB | 0.99 | 4.1 |
| Random Forest (top 25) | 25 | 0.878 | 87.9% | 3,796 KB | 55.60 | 3.2 |

*Size = compressed joblib file. Latency = median scikit-learn prediction on a laptop/cloud CPU — relative
comparison only, not a wearable measurement. Timings vary by machine; `train.py` regenerates them.*

**Recommendation: Logistic Regression on the full feature set.**
- Highest quality on new users (Macro F1 0.966; per-user accuracy 91.9 %–99.5 %); grouped 4-fold CV on the
  training users: 0.942 ± 0.014.
- Smallest and simplest model: 25 KB, ~7,900 arithmetic operations per prediction (≈5 µs as plain NumPy),
  against a budget of one prediction every 1.28 s.
- Fully interpretable: one weight per feature and activity.
- Remaining errors are mostly *sitting ↔ standing* (58 of 82 errors), which look almost identical to a waist-worn sensor.
- The 25-feature model cuts on-device feature computation by 95.5 % but makes ~2.8× more errors — an option only
  for the most constrained hardware.

## Business value

- **Day-one accuracy for new customers** — validated only on people never used for training (no per-user calibration).
- **Raw data stays on the device** — uploading activity summaries instead of raw 50 Hz sensor streams reduces upload
  volume by >99 % in the app's illustrative scenario; the *Business case* page lets you change every assumption
  (fleet size, wear time, cloud prices) and see the cloud-cost impact.
- **Negligible compute and memory** for the classifier itself, leaving the hardware budget for feature extraction.
- **Explainable** behaviour that product, QA and regulatory teams can audit.

## Method

1. **Data:** UCI HAR — 10,299 windows (2.56 s, 50 % overlap) × 561 engineered accelerometer/gyroscope features, 6 activities, 30 subjects.
2. **Leakage control:** subjects 25–30 held out completely; StratifiedGroupKFold (4 folds) on subjects 1–24 for
   all selection decisions; feature ranking learned on training subjects only.
3. **Candidates:** Logistic Regression (with scaling), LightGBM, Random Forest — each with all 561 features and with
   the top-25 features by Random Forest importance; feature-budget curve from 5 to 561 features (grouped CV).
4. **Evaluation:** Macro F1 (primary), accuracy, per-class and per-user analysis, confusion matrices.
5. **Edge benchmark:** compressed model size, median single-window and 1,000-window CPU latency after warm-up.

## Limitations & next steps

- Single test split with six users; lab data from a waist-mounted phone (a wrist device in daily life differs).
- The cost of computing 561 features on a microcontroller was not measured — the main engineering risk.
- Next: leave-users-out CV over all 30 subjects · sparse (L1) Logistic Regression for fewer features ·
  port to C / TFLite Micro and benchmark on target hardware including energy · collect wrist-worn, free-living data.

## Run it yourself

```bash
pip install -r requirements.txt
python train.py          # downloads UCI HAR, trains, benchmarks, writes ./artifacts (~3–5 min)
streamlit run app.py
```

`train.py --data-dir "path/to/UCI HAR Dataset"` uses a local copy instead of downloading.

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository (the `artifacts/` folder must be committed — the app does not train at start-up).
2. On [share.streamlit.io](https://share.streamlit.io) choose *Create app*, select the repository, main file `app.py`,
   and Python 3.13 under *Advanced settings*.
3. Replace `GITHUB_URL` at the top of `app.py` and the live-app link above with your own URLs.

## Project structure

```
app.py                  Streamlit app (reads artifacts only)
train.py                data download, training, evaluation, benchmarking, artifact export
artifacts/metrics.json  all results shown in the app
artifacts/models/       Logistic Regression (full, top-25) and LightGBM pipelines
artifacts/test_windows.npz  sensor windows of the unseen test users for the live demo
requirements.txt        pinned dependencies (must match the versions used to save the models)
```

## Data & licence

Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). *Human Activity Recognition Using
Smartphones* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K — licensed CC BY 4.0.
"EdgeFit" is a fictional company used to frame the case study.
