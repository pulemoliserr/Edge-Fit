"""
EdgeFit — On-device Human Activity Recognition (portfolio app)

Every number on these pages is read from artifacts/metrics.json, which is produced by
train.py on the real UCI HAR dataset (subjects 1–24 train, unseen subjects 25–30 test).
Business estimates are computed live from assumptions the visitor can change.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ART = Path(__file__).parent / "artifacts"
GITHUB_URL = "https://github.com/your-username/edgefit-har"   # <- replace with your repository

st.set_page_config(page_title="EdgeFit · On-Device Activity Recognition", page_icon="⌚",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 3.5rem; max-width: 1250px;}
.kicker {font-size: .8rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #2563EB;}
.lead {font-size: 1.08rem; color: #475569; margin-bottom: 1.2rem;}
div[data-testid="stMetric"] {background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 16px;}
.note {font-size: .85rem; color: #64748B;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data
@st.cache_data
def load_metrics():
    path = ART / "metrics.json"
    if not path.exists():
        st.error("artifacts/metrics.json not found. Run `python train.py` first.")
        st.stop()
    return json.loads(path.read_text())


@st.cache_resource
def load_models():
    return {
        "Logistic Regression (561 features)": joblib.load(ART / "models" / "lr_full.joblib"),
        f"Logistic Regression (top {M['edge_k']})": joblib.load(ART / "models" / f"lr_top{M['edge_k']}.joblib"),
        "LightGBM (561 features)": joblib.load(ART / "models" / "lgbm_full.joblib"),
    }


@st.cache_data
def load_windows():
    d = np.load(ART / "test_windows.npz")
    return d["X"], d["y"], d["subject"]


M = load_metrics()
ACT = M["activities"]
ACT_NICE = [a.replace("_", " ").title() for a in ACT]
R = pd.DataFrame(M["results"]).sort_values("test_macro_f1", ascending=False).reset_index(drop=True)
LR = R.loc[R.model == "Logistic Regression"].iloc[0]
LR_K = R.loc[R.model == f"Logistic Regression (top {M['edge_k']})"].iloc[0]
DS = M["dataset"]
PRED_EVERY_S = DS["window_seconds"] * (1 - DS["overlap"])     # a new prediction every 1.28 s
per_subj = pd.DataFrame(M["per_subject"])

# ----------------------------------------------------------------------------- sidebar
st.sidebar.markdown("## ⌚ EdgeFit")
st.sidebar.caption("On-device human activity recognition for wearables")
page = st.sidebar.radio("Navigate", ["Executive summary", "Business case", "Model benchmark",
                                     "Edge optimisation", "Live demo", "Method & limitations"])
st.sidebar.divider()
st.sidebar.markdown(
    f"**Data:** UCI HAR (real sensor data)  \n"
    f"**Validation:** {DS['subjects'] - len(DS['test_subjects'])} train subjects, "
    f"{len(DS['test_subjects'])} unseen test users  \n"
    f"**Stack:** Python · scikit-learn · LightGBM · Streamlit")
st.sidebar.markdown(f"[Source code on GitHub]({GITHUB_URL})")
st.sidebar.caption(f"Results generated {M['generated_utc']}")


def fmt_pct(x, d=1):
    return f"{x * 100:.{d}f}%"


# ============================================================================= pages
if page == "Executive summary":
    st.markdown('<div class="kicker">Case study · Edge AI · Wearables</div>', unsafe_allow_html=True)
    st.title("Recognising activities on the device, not in the cloud")
    st.markdown(
        f'<div class="lead">A wearable maker wants to recognise what a user is doing — walking, climbing stairs, '
        f'sitting, standing, lying — directly on a low-cost device, for people the model has never seen. '
        f'I compared six model candidates on real smartphone sensor data from {DS["subjects"]} people and '
        f'recommend a model that is accurate, tiny and fully explainable.</div>', unsafe_allow_html=True)

    c = st.columns(4)
    c[0].metric("Macro F1 on unseen users", f"{LR.test_macro_f1:.3f}",
                help="Average F1 over the six activities, measured once on subjects 25–30 who were never used for training or tuning.")
    c[1].metric("Accuracy on unseen users", fmt_pct(LR.test_accuracy),
                help=f"{DS['test_windows']:,} test windows from {len(DS['test_subjects'])} new users.")
    c[2].metric("Model size", f"{LR.size_mb * 1024:.0f} KB",
                help="Compressed joblib file of the full pipeline (scaler + model).")
    c[3].metric("Compute per prediction", f"{M['native_lr_latency_us']:.0f} µs",
                help=f"Measured as plain NumPy maths on a laptop CPU: about {M['lr_ops_per_window']:,} arithmetic operations per window. "
                     f"A new prediction is needed only every {PRED_EVERY_S:.2f} s.")

    st.subheader("What this means for the business")
    b = st.columns(3)
    b[0].markdown(f"**Works for new customers on day one.** The model was tested only on people it never saw. "
                  f"Accuracy per new user ranged from **{fmt_pct(per_subj.accuracy.min())}** to "
                  f"**{fmt_pct(per_subj.accuracy.max())}** — no per-user calibration needed.")
    b[1].markdown(f"**Fits on almost any wearable.** At ~{LR.size_mb * 1024:.0f} KB and ~{M['lr_ops_per_window']:,} "
                  f"operations per prediction, the classifier itself is negligible next to a "
                  f"{PRED_EVERY_S:.2f}-second prediction budget. Raw sensor data can stay on the device "
                  f"(see *Business case*).")
    b[2].markdown("**Explainable and auditable.** One weight per sensor feature and activity — product, QA and "
                  "regulatory teams can see why an activity was predicted, unlike a black-box network.")

    st.subheader("Project in brief (STAR)")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"""
**Situation** — Activity tracking that streams raw motion data to the cloud costs bandwidth, cloud spend and
privacy exposure. On-device recognition avoids that, but the model must be small, fast and reliable for new users.

**Task** — Recommend a model for *previously unseen users* by balancing predictive quality, latency, model size,
feature burden and interpretability — with no data leakage.
""")
    with s2:
        best_tree = R[R.family.isin(["LightGBM", "Random Forest"]) & (R.features == DS["features"])].iloc[0]
        st.markdown(f"""
**Action** — Subject-level holdout (users 25–30 never touched), 4-fold grouped cross-validation on users 1–24,
six candidates (linear, gradient boosting, random forest; full vs. top-{M['edge_k']} features), CPU latency and
size benchmarks, error analysis per activity and per user.

**Result** — Logistic Regression won on every axis that matters here: Macro F1 **{LR.test_macro_f1:.3f}**
vs. **{best_tree.test_macro_f1:.3f}** for the best tree model ({best_tree.model}), at a fraction of the size.
The remaining errors are mostly *sitting ↔ standing*, which look almost identical to a waist-worn sensor.
""")

    st.subheader("All candidates at a glance")
    show = R[["model", "features", "test_macro_f1", "test_accuracy", "size_mb", "latency_single_ms", "train_time_s"]].rename(
        columns={"model": "Model", "features": "Features", "test_macro_f1": "Macro F1", "test_accuracy": "Accuracy",
                 "size_mb": "Size (MB)", "latency_single_ms": "Latency / window (ms)", "train_time_s": "Train time (s)"})
    st.dataframe(show.style.format({"Macro F1": "{:.3f}", "Accuracy": "{:.1%}", "Size (MB)": "{:.3f}",
                                    "Latency / window (ms)": "{:.2f}", "Train time (s)": "{:.1f}"})
                 .highlight_max(subset=["Macro F1", "Accuracy"], color="#D1FAE5")
                 .highlight_min(subset=["Size (MB)", "Latency / window (ms)"], color="#DBEAFE"),
                 hide_index=True, width="stretch")
    st.markdown('<div class="note">Latencies are scikit-learn calls on a laptop CPU (including framework overhead); '
                'they compare models with each other and are not wearable measurements.</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- business case
elif page == "Business case":
    st.markdown('<div class="kicker">Business case</div>', unsafe_allow_html=True)
    st.title("Cloud streaming vs. on-device recognition")
    st.markdown('<div class="lead">An interactive estimate. Change the assumptions on the left — every number on '
                'this page is recalculated from them. The defaults are illustrative, not quotes from a provider.</div>',
                unsafe_allow_html=True)

    a, r = st.columns([1, 2])
    with a:
        st.markdown("##### Assumptions")
        fleet = st.number_input("Active devices", 1_000, 50_000_000, 100_000, step=10_000)
        hours = st.slider("Hours worn per day", 1, 24, 14)
        hz = st.number_input("Sensor sampling rate (Hz)", 10, 400, DS["sampling_hz"])
        channels = st.number_input("Sensor channels (3-axis accel + 3-axis gyro)", 1, 12, 6)
        bytes_per = st.selectbox("Bytes per sample value", [2, 4], index=0, help="2 = 16-bit IMU reading, 4 = float32")
        cost_gb = st.number_input("Cloud cost per GB ingested, stored & processed (USD)", 0.0, 5.0, 0.10, step=0.01)
        cost_m_inf = st.number_input("Cloud cost per 1M model predictions (USD)", 0.0, 50.0, 0.50, step=0.10)
        summary_bytes = st.number_input("On-device upload per minute (bytes)", 1, 10_000, 64,
                                        help="E.g. an activity summary per minute: minutes per activity, step counts.")

    secs = hours * 3600
    raw_day = hz * channels * bytes_per * secs                       # bytes per device per day
    edge_day = summary_bytes * hours * 60
    preds_day = secs / PRED_EVERY_S
    month = 30
    raw_gb_month = raw_day * fleet * month / 1e9
    edge_gb_month = edge_day * fleet * month / 1e9
    data_cost_cloud = raw_gb_month * cost_gb
    infer_cost_cloud = preds_day * fleet * month / 1e6 * cost_m_inf
    cloud_cost = data_cost_cloud + infer_cost_cloud
    edge_cost = edge_gb_month * cost_gb
    err_full = 1 - LR.test_accuracy
    err_k = 1 - LR_K.test_accuracy

    with r:
        k = st.columns(3)
        k[0].metric("Upload per device/day", f"{edge_day / 1e3:,.1f} KB",
                    help=f"On-device summaries, compared with {raw_day / 1e6:,.1f} MB of raw sensor data per day.")
        k[1].metric("Data volume reduction", f"{(1 - edge_day / raw_day) * 100:.2f}%",
                    help="Share of upload volume avoided by not streaming raw sensor data.")
        k[2].metric("Cost avoided/month", f"${cloud_cost - edge_cost:,.0f}",
                    help=f"For {fleet:,} devices under the assumptions on the left.")
        st.caption(f"Raw streaming: {raw_day / 1e6:,.1f} MB per device per day · "
                   f"{preds_day:,.0f} predictions per device per day (one every {PRED_EVERY_S:.2f} s).")

        comp = pd.DataFrame({
            "Scenario": ["Stream raw data,<br>classify in cloud"] * 2 + ["Classify on device,<br>upload summaries"] * 2,
            "Cost component": ["Data ingest & storage", "Cloud inference"] * 2,
            "USD per month": [data_cost_cloud, infer_cost_cloud, edge_cost, 0.0]})
        fig = px.bar(comp, x="Scenario", y="USD per month", color="Cost component", text_auto=",.0f",
                     color_discrete_map={"Data ingest & storage": "#94A3B8", "Cloud inference": "#2563EB"})
        fig.update_layout(height=340, margin=dict(t=20, b=10), xaxis_title=None,
                          yaxis_title="Est. cloud cost per month (USD)", legend=dict(orientation="h", y=1.12, title=None))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame({
            "Scenario": ["Stream raw data, classify in cloud", "Classify on device, upload summaries"],
            "Data per month (TB)": [raw_gb_month / 1e3, edge_gb_month / 1e3],
            "Est. cloud cost per month (USD)": [cloud_cost, edge_cost]})
            .style.format({"Data per month (TB)": "{:,.2f}", "Est. cloud cost per month (USD)": "${:,.0f}"}),
            hide_index=True, width="stretch")

        st.markdown("##### What accuracy means for users")
        u = st.columns(2)
        u[0].metric("Mislabelled time/day — full model", f"{err_full * hours * 60:.0f} min",
                    help=f"Error rate {fmt_pct(err_full)} on unseen users × hours worn.")
        u[1].metric(f"Mislabelled time/day — {M['edge_k']} features", f"{err_k * hours * 60:.0f} min",
                    f"{err_k / err_full:.1f}× more errors", delta_color="inverse",
                    help=f"Error rate {fmt_pct(err_k)} on unseen users × hours worn.")
        st.markdown(
            f'<div class="note">Not counted here: battery savings from keeping the radio off, privacy/compliance benefits '
            f'of not transmitting raw motion data, and the engineering cost of computing features on the device. '
            f'Errors are concentrated in sitting ↔ standing; walking, stairs and lying are recognised with F1 ≥ 0.97.</div>',
            unsafe_allow_html=True)


# ----------------------------------------------------------------------------- benchmark
elif page == "Model benchmark":
    st.markdown('<div class="kicker">Model benchmark</div>', unsafe_allow_html=True)
    st.title("Quality vs. cost of every candidate")
    st.markdown(f'<div class="lead">All models were trained on users 1–24 and evaluated once on the '
                f'{DS["test_windows"]:,} windows of unseen users 25–30. Bubble size = model file size.</div>',
                unsafe_allow_html=True)

    plot_df = R.assign(bubble=R.size_mb.clip(lower=0.35))   # keep the KB-sized models visible
    fig = px.scatter(plot_df, x="latency_single_ms", y="test_macro_f1", size="bubble", color="family", text="model",
                     log_x=True, size_max=40,
                     hover_data={"features": True, "test_accuracy": ":.3f", "size_mb": ":.3f", "bubble": False},
                     labels={"latency_single_ms": "CPU latency per window (ms, log scale)", "test_macro_f1": "Macro F1 (unseen users)",
                             "family": "Model family"},
                     color_discrete_map={"Logistic Regression": "#2563EB", "LightGBM": "#F59E0B", "Random Forest": "#10B981"})
    fig.update_traces(textposition="middle right", textfont_size=11)
    lat = np.log10(R.latency_single_ms)
    fig.update_layout(height=470, margin=dict(t=20), xaxis_range=[lat.min() - 0.3, lat.max() + 0.8],
                      yaxis_range=[R.test_macro_f1.min() - 0.015, R.test_macro_f1.max() + 0.015])
    st.plotly_chart(fig, width="stretch")
    st.caption("Upper-left is better: high quality, low latency. Bubbles are drawn with a minimum size so the "
               "KB-sized Logistic Regression models stay visible — exact sizes are in the hover and the overview table.")

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Where does a model make mistakes?")
        sel = st.selectbox("Model", R.model.tolist())
        cm = np.array(M["confusion"][sel])
        cm_pct = cm / cm.sum(axis=1, keepdims=True)
        heat = go.Figure(go.Heatmap(z=cm_pct, x=ACT_NICE, y=ACT_NICE, colorscale="Blues", zmin=0, zmax=1,
                                    text=cm, texttemplate="%{text}", hovertemplate="True %{y}<br>Predicted %{x}<br>%{z:.1%} of row<extra></extra>"))
        heat.update_layout(height=430, yaxis_autorange="reversed", margin=dict(t=10),
                           xaxis_title="Predicted", yaxis_title="True activity")
        st.plotly_chart(heat, width="stretch")
    with c2:
        st.subheader("F1 per activity")
        rep = M["reports"][sel]
        f1 = pd.DataFrame({"Activity": ACT_NICE, "F1": [rep[a]["f1-score"] for a in ACT]})
        bar = px.bar(f1, x="F1", y="Activity", orientation="h", text_auto=".3f", range_x=[0.5, 1],
                     color_discrete_sequence=["#2563EB"])
        bar.update_layout(height=300, margin=dict(t=10), yaxis_title=None)
        st.plotly_chart(bar, width="stretch")
        st.markdown("**Reading:** static postures *sitting* and *standing* are the hardest pair for every model — "
                    "both are upright with almost no motion. Dynamic activities and lying are near-perfect.")

    st.subheader("Does it work for every new user?")
    ps = per_subj.assign(Subject=lambda d: "User " + d.subject.astype(str))
    pf = px.bar(ps, x="Subject", y="accuracy", text_auto=".1%", range_y=[0.8, 1], color_discrete_sequence=["#2563EB"],
                labels={"accuracy": "Accuracy (Logistic Regression)"})
    pf.add_hline(y=LR.test_accuracy, line_dash="dash", annotation_text=f"overall {fmt_pct(LR.test_accuracy)}")
    pf.update_layout(height=320, margin=dict(t=20), xaxis_title=None)
    st.plotly_chart(pf, width="stretch")
    st.caption(f"Grouped 4-fold cross-validation on the training users gave Macro F1 {M['lr_grouped_cv']['mean']:.3f} "
               f"± {M['lr_grouped_cv']['std']:.3f}; the spread between users shows why a subject-level split matters.")


# ----------------------------------------------------------------------------- edge
elif page == "Edge optimisation":
    st.markdown('<div class="kicker">Edge optimisation</div>', unsafe_allow_html=True)
    st.title("How many sensor features does the device really need?")
    st.markdown('<div class="lead">Each feature must be computed on the wearable for every window, so fewer features '
                'mean less on-device work. Features were ranked by Random Forest importance on the training users only.</div>',
                unsafe_allow_html=True)

    cur = pd.DataFrame(M["feature_curve"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cur.features, y=cur.cv_macro_f1_mean, mode="lines+markers", name="Grouped-CV Macro F1",
                             line=dict(color="#2563EB", width=3),
                             error_y=dict(type="data", array=cur.cv_macro_f1_std, visible=True, color="#93C5FD")))
    fig.update_layout(height=420, xaxis=dict(type="log", title="Number of input features (log scale)",
                                             tickvals=cur.features.tolist()),
                      yaxis=dict(title="Macro F1 (grouped CV, users 1–24)"), margin=dict(t=20))
    edge_pt = cur[cur.features == M["edge_k"]]
    fig.add_trace(go.Scatter(x=edge_pt.features, y=edge_pt.cv_macro_f1_mean, mode="markers+text", name="Edge model",
                             marker=dict(size=16, color="#F59E0B", line=dict(width=2, color="white")),
                             text=[f"{M['edge_k']} features"], textposition="bottom right"))
    fig.update_layout(legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")

    c25 = cur.loc[cur.features == M["edge_k"], "cv_macro_f1_mean"].item()
    cfull = cur.loc[cur.features == DS["features"], "cv_macro_f1_mean"].item()
    st.markdown(f"Quality rises steeply up to about 15 features, then flattens: **{M['edge_k']} features keep "
                f"{c25 / cfull:.0%} of the full model's cross-validated Macro F1** with "
                f"{1 - M['edge_k'] / DS['features']:.1%} fewer inputs.")

    st.subheader(f"Full model vs. {M['edge_k']}-feature edge model (unseen users)")
    t = pd.DataFrame({
        "": ["Input features to compute", "Macro F1", "Accuracy", "Model size", "Mis-labelled windows per 1,000"],
        "Full (561)": [f"{DS['features']}", f"{LR.test_macro_f1:.3f}", fmt_pct(LR.test_accuracy),
                       f"{LR.size_mb * 1024:.1f} KB", f"{(1 - LR.test_accuracy) * 1000:.0f}"],
        f"Edge ({M['edge_k']})": [f"{M['edge_k']}", f"{LR_K.test_macro_f1:.3f}", fmt_pct(LR_K.test_accuracy),
                                  f"{LR_K.size_mb * 1024:.1f} KB", f"{(1 - LR_K.test_accuracy) * 1000:.0f}"]})
    st.dataframe(t, hide_index=True, width="stretch")
    st.info(f"**Recommendation:** ship the full {DS['features']}-feature model where the device can compute the "
            f"standard feature set; it is still only ~{LR.size_mb * 1024:.0f} KB. Use the {M['edge_k']}-feature model "
            f"only for the most constrained hardware, accepting about "
            f"{(1 - LR_K.test_accuracy) / (1 - LR.test_accuracy):.1f}× more errors.")

    st.subheader(f"The {M['edge_k']} selected features")
    tf = pd.DataFrame(M["top_features"])
    tf["group"] = np.select([tf.feature.str.contains("Gravity|angle", case=False), tf.feature.str.startswith("f")],
                            ["Gravity / orientation", "Frequency domain"], "Body motion (time domain)")
    ff = px.bar(tf.sort_values("importance"), x="importance", y="feature", color="group", orientation="h",
                labels={"importance": "Random Forest importance (training users)", "feature": ""},
                color_discrete_sequence=["#2563EB", "#F59E0B", "#10B981"])
    ff.update_layout(height=620, margin=dict(t=10), legend_title=None)
    st.plotly_chart(ff, width="stretch")
    st.caption("Gravity/orientation features dominate: the direction of gravity relative to the device separates lying, "
               "sitting and standing. Many of them are highly correlated, so the ranking shows usefulness, not causality.")


# ----------------------------------------------------------------------------- live demo
elif page == "Live demo":
    st.markdown('<div class="kicker">Live demo</div>', unsafe_allow_html=True)
    st.title("Classify real sensor windows from unseen users")
    st.markdown(f'<div class="lead">Pick a real {DS["window_seconds"]}-second window recorded from one of the test users. '
                f'The models below have never seen this person.</div>', unsafe_allow_html=True)
    X, y, subj = load_windows()
    models = load_models()

    c1, c2, c3 = st.columns([1, 1, 1])
    user = c1.selectbox("Test user", sorted(np.unique(subj).tolist()), format_func=lambda s: f"User {s}")
    true_filter = c2.selectbox("Activity (ground truth)", ["Any"] + ACT_NICE)
    pool = np.where((subj == user) & ((y == ACT_NICE.index(true_filter)) if true_filter != "Any" else True))[0]
    if "win" not in st.session_state or st.session_state.get("pool_key") != (user, true_filter):
        st.session_state.win = int(np.random.default_rng().choice(pool))
        st.session_state.pool_key = (user, true_filter)
    if c3.button("🎲 Random window", width="stretch"):
        st.session_state.win = int(np.random.default_rng().choice(pool))
    i = st.session_state.win
    x = X[i:i + 1]
    st.markdown(f"**Window #{i}** · User {subj[i]} · true activity: **{ACT_NICE[y[i]]}**")

    rows, probs = [], {}
    k_idx = np.array(M["top_k_indices"])
    for name, model in models.items():
        xin = x[:, k_idx] if "top" in name else x
        t0 = time.perf_counter_ns()
        p = model.predict_proba(xin)[0]
        ms = (time.perf_counter_ns() - t0) / 1e6
        probs[name] = p
        rows.append({"Model": name, "Prediction": ACT_NICE[int(np.argmax(p))], "Confidence": p.max(),
                     "Correct": "✅" if int(np.argmax(p)) == y[i] else "❌", "Latency (ms)": ms})
    st.dataframe(pd.DataFrame(rows).style.format({"Confidence": "{:.1%}", "Latency (ms)": "{:.2f}"}),
                 hide_index=True, width="stretch")

    pdf = pd.DataFrame([{"Model": n, "Activity": ACT_NICE[j], "Probability": float(p[j])}
                        for n, p in probs.items() for j in range(6)])
    fig = px.bar(pdf, x="Activity", y="Probability", color="Model", barmode="group", range_y=[0, 1],
                 color_discrete_sequence=["#2563EB", "#93C5FD", "#F59E0B"])
    fig.update_layout(height=380, margin=dict(t=20), xaxis_title=None, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption("Latency here includes Python/scikit-learn overhead on a shared cloud server. The same Logistic Regression "
               f"written as plain arithmetic took about {M['native_lr_latency_us']:.0f} µs per window in the benchmark.")


# ----------------------------------------------------------------------------- method
elif page == "Method & limitations":
    st.markdown('<div class="kicker">Method & limitations</div>', unsafe_allow_html=True)
    st.title("How the results were produced — and what they don't show")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
#### Data
- **UCI HAR** (DOI {DS['doi']}): {DS['windows']:,} windows of {DS['window_seconds']} s ({int(DS['overlap'] * 100)} % overlap),
  {DS['features']} engineered accelerometer/gyroscope features, {DS['subjects']} volunteers wearing a smartphone on the waist.
- Six activities, mildly imbalanced (largest/smallest class ≈ 1.4×).

#### Leakage control
- **Subject-level holdout:** users {DS['test_subjects'][0]}–{DS['test_subjects'][-1]} are never used for training,
  feature selection or tuning — they simulate new customers.
- **Grouped cross-validation** (StratifiedGroupKFold, 4 folds) on users 1–24 for all model-selection decisions.
- Feature ranking learned on training users only.

#### Metrics
- **Macro F1** as primary metric (every activity counts equally), accuracy as secondary.
- Model size = compressed joblib file; latency = median of repeated single-window CPU predictions after warm-up.
""")
    with c2:
        env = M["environment"]
        st.markdown(f"""
#### Limitations
- **One test split with six users** — per-user accuracy varies between {fmt_pct(per_subj.accuracy.min())}
  and {fmt_pct(per_subj.accuracy.max())}; more users would tighten the estimate.
- **Lab data, phone on the waist** — a wrist-worn device in daily life will look different and needs new data.
- **Feature extraction cost is not measured** — computing 561 features (incl. FFTs) on a microcontroller is
  the main remaining engineering risk.
- **Latency measured on a laptop/cloud CPU**, not on wearable hardware. Business estimates use editable assumptions.

#### Next steps
1. Leave-users-out cross-validation over all 30 subjects.
2. Sparse (L1) Logistic Regression to cut features without the accuracy drop of the top-{M['edge_k']} model.
3. Port to C / TensorFlow Lite Micro and benchmark on target hardware, including feature extraction and energy.
4. Collect wrist-worn, free-living data and track drift after launch.

<div class="note">Benchmark environment: {env['platform']} · Python {env['python']} ·
scikit-learn {env['scikit_learn']} · LightGBM {env['lightgbm']}</div>
""", unsafe_allow_html=True)
    st.caption("Dataset: D. Anguita, A. Ghio, L. Oneto, X. Parra, J. L. Reyes-Ortiz (2013), Human Activity Recognition "
               "Using Smartphones, UCI Machine Learning Repository, CC BY 4.0.")
