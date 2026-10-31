# EdgeFit: On-Device Human Activity Recognition for Wearables

**Author:** [Your Name] · **Domain:** Wearables, Edge AI, HealthTech
**Tech stack:** Python, scikit-learn, LightGBM, Pandas, NumPy, Plotly, Streamlit, Joblib
**Links:** [Live app](https://your-app.streamlit.app) · [GitHub](https://github.com/your-username/edgefit-har)

> Case study built on the public UCI HAR dataset (30 people, real smartphone sensor data).
> "EdgeFit" is a fictional wearable company used to frame the business problem.

---

## Headline results — measured on 6 users the model never saw

| | |
|---|---|
| **Macro F1** | **0.966** (accuracy 96.4 %) |
| **Accuracy per new user** | 91.9 % – 99.5 % |
| **Model size** | **25 KB** |
| **Compute per prediction** | ~7,900 operations (≈ 5 µs on a laptop CPU) vs. a 1.28 s budget |
| **Feature-light variant** | 95.5 % fewer input features, Macro F1 0.897 |

## Situation
Wearables that stream raw motion data to the cloud for activity recognition pay for it in bandwidth, cloud
spend and privacy exposure. Recognising activities **on the device** avoids that — but the model must be small,
fast, explainable and accurate for **new customers** from day one.

## Task
Recommend a deployable model for previously unseen users, balancing predictive quality, model size, CPU latency,
on-device feature burden and interpretability — without data leakage.

## Action
- **Leakage-safe validation:** held out users 25–30 completely; used 4-fold StratifiedGroupKFold on users 1–24
  for every modelling decision, so no person appears in both training and evaluation.
- **Six candidates:** Logistic Regression, LightGBM and Random Forest, each with all 561 engineered sensor
  features and with a top-25 subset selected from Random Forest importance on training users only.
- **Edge trade-off curve:** cross-validated quality for 5 to 561 features to quantify what each feature budget costs.
- **Edge benchmark:** compressed model size and median single-window / batch CPU latency.
- **Error analysis:** per-activity F1, confusion matrices and per-user accuracy.
- **Streamlit app:** executive summary, interactive cloud-vs-device business-case calculator, model benchmark,
  feature-budget explorer and a live demo that classifies real sensor windows from unseen users.

## Result
| Model | Features | Macro F1 | Accuracy | Size |
|---|---:|---:|---:|---:|
| **Logistic Regression** | 561 | **0.966** | **96.4 %** | **25 KB** |
| LightGBM | 561 | 0.943 | 94.1 % | 1.7 MB |
| Random Forest | 561 | 0.927 | 92.8 % | 4.1 MB |
| Logistic Regression (top 25) | 25 | 0.897 | 89.9 % | 2 KB |
| LightGBM (top 25) | 25 | 0.883 | 88.1 % | 2.3 MB |
| Random Forest (top 25) | 25 | 0.878 | 87.9 % | 3.7 MB |

- **Recommended: Logistic Regression** — the most accurate *and* the smallest, fastest and most explainable
  candidate; more complex models did not generalise better to new users.
- Remaining errors concentrate on **sitting vs. standing** (58 of 82 errors) — both upright, almost motionless postures.
- The **25-feature** model cuts on-device feature computation by 95.5 % but makes about 2.8× more errors,
  so it is reserved for the most constrained hardware.

## Business impact
- **Works for new customers immediately** — validated exclusively on people outside the training data.
- **Raw sensor data never has to leave the device:** in the app's illustrative scenario (100k devices, 14 h/day,
  50 Hz × 6 channels), uploading activity summaries instead of raw streams cuts upload volume by over 99 %;
  visitors can change every assumption and see the cloud-cost effect.
- **Tiny footprint** (25 KB, a few thousand operations per prediction) leaves the device's compute budget for
  sensor processing.
- **Explainable decisions** — one weight per feature and activity — support QA, debugging and regulatory review.

## What I would do next
Leave-users-out validation across all 30 people · sparse (L1) Logistic Regression to reduce features without
the accuracy loss · port to C / TensorFlow Lite Micro and measure latency and energy on real wearable hardware,
including feature extraction · collect wrist-worn, free-living data and monitor drift after launch.

## Skills demonstrated
Leakage-aware validation (grouped CV, subject holdout) · model selection under multiple constraints ·
feature selection · performance benchmarking · error analysis · translating ML metrics into business terms ·
interactive data apps (Streamlit, Plotly) · reproducible pipelines (pinned dependencies, single training script).
