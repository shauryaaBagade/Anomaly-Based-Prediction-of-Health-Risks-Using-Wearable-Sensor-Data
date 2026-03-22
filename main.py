"""
============================================================
Anomaly-Based Prediction of Health Risks Using Wearable Sensor Data
============================================================
Files  : activity_environment_data.csv
         digital_interaction_data.csv
         personal_health_data.csv
Labels : Anomaly_Flag (ground truth)

Models:
  1. Isolation Forest       — unsupervised baseline
  2. One-Class SVM          — unsupervised (better boundary)
  3. Autoencoder            — unsupervised deep learning main model
  4. Random Forest          — supervised upper bound
  5. Gradient Boosting      — supervised strongest model

WHY ONE-CLASS SVM?
------------------
Your Anomaly_Flag reflects a user's HEALTH PROFILE (chronic
conditions, BMI, age) not just sensor spikes. This means normal
vs anomaly samples have similar raw sensor values but live in
different regions of the feature space. One-Class SVM learns a
tight boundary around the normal class in a kernel-transformed
space — it handles this kind of "profile-based" anomaly much
better than reconstruction error or isolation depth.

KEY IMPROVEMENTS:
• PCA-reduced features fed to Autoencoder (removes noise,
  preserves 95% of variance → cleaner reconstruction signal)
• One-Class SVM added as second unsupervised model
• Richer health-context features (BMI, age group, medication)
• All models use macro-F1 threshold tuning
============================================================
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import (IsolationForest, RandomForestClassifier,
                               HistGradientBoostingClassifier)
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

print("=" * 60)
print("  Health-Risk Anomaly Detection – Wearable Sensor Data")
print("=" * 60)


# ╔══════════════════════════════════════════════════════════╗
# ║  1. LOAD & MERGE                                        ║
# ╚══════════════════════════════════════════════════════════╝

def load_and_merge():
    print("\n[1/7] Loading and merging CSV files ...")
    act  = pd.read_csv("activity_environment_data.csv")
    dig  = pd.read_csv("digital_interaction_data.csv")
    hlth = pd.read_csv("personal_health_data.csv")
    for df in [act, dig, hlth]:
        df.columns = [c.strip().lower() for c in df.columns]
    df = pd.merge(act, dig,  on=["user_id", "timestamp"], how="inner")
    df = pd.merge(df,  hlth, on=["user_id", "timestamp"], how="inner")
    print(f"      Merged shape : {df.shape}")
    print(f"      Anomaly_Flag : {df['anomaly_flag'].value_counts().to_dict()}")
    return df


# ╔══════════════════════════════════════════════════════════╗
# ║  2. PREPROCESS + FEATURE ENGINEERING                    ║
# ╚══════════════════════════════════════════════════════════╝

STRESS_MAP = {"Low": 0, "Medium": 1, "High": 2}
MOOD_MAP   = {"Happy": 0, "Neutral": 1, "Sad": 2,
              "Anxious": 3, "Stressed": 4}
ECG_MAP    = {"Normal": 0, "Abnormal": 1, "Borderline": 0.5}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Encode categoricals ───────────────────────────────
    df["stress_num"]    = df["stress_level"].map(STRESS_MAP).fillna(1)
    df["mood_num"]      = df["mood"].map(MOOD_MAP).fillna(1)
    df["ecg_num"]       = df["ecg"].map(ECG_MAP).fillna(0)
    df["smoker_num"]    = (df["smoker"].str.lower() == "yes").astype(int)
    df["medication_num"]= (df["medication"].str.lower() != "none").astype(int)
    df["snoring_num"]   = (df["snoring"].str.lower() == "yes").astype(int)

    # ── BMI from weight/height ────────────────────────────
    if "weight" in df.columns and "height" in df.columns:
        df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2 + 1e-9)
    else:
        df["bmi"] = 22.0  # default healthy BMI if missing

    # ── Age group risk (higher age = higher baseline risk) ─
    if "age" in df.columns:
        df["age_risk"] = pd.cut(
            df["age"],
            bins=[0, 30, 45, 60, 200],
            labels=[0, 1, 2, 3]
        ).astype(float).fillna(0)
    else:
        df["age_risk"] = 0

    # ── Composite clinical indices ────────────────────────

    # 1. Cardiovascular stress index
    hr_z   = (df["heart_rate"] - df["heart_rate"].mean()) / (df["heart_rate"].std() + 1e-9)
    spo2_z = -(df["blood_oxygen_level"] - df["blood_oxygen_level"].mean()) / (df["blood_oxygen_level"].std() + 1e-9)
    df["cardio_index"] = hr_z + spo2_z + df["stress_num"] + df["ecg_num"] * 2

    # 2. Sleep quality index (higher = worse)
    df["sleep_index"] = (
        df["wakeups"] * 2
        + df["snoring_num"]
        - df["deep_sleep_duration"].fillna(0)
        - df["rem_sleep_duration"].fillna(0)
    )

    # 3. Lifestyle risk (sedentary + smoking + alcohol)
    df["alcohol_num"] = df.get("alcohol_consumption", pd.Series(["None"]*len(df))).map(
        {"None": 0, "Low": 1, "Moderate": 2, "High": 3}
    ).fillna(0)
    df["lifestyle_risk"] = (
        (1 - df["steps"] / (df["steps"].max() + 1)) * 3
        + df["smoker_num"] * 2
        + df["alcohol_num"]
        + df["body_fat_percentage"] / (df["body_fat_percentage"].max() + 1)
    )

    # 4. Digital-mental stress
    df["digital_stress"] = (
        df["screen_time"] / (df["screen_time"].max() + 1)
        + df["notifications_received"] / (df["notifications_received"].max() + 1)
        + df["stress_num"] / 2
        + df["mood_num"] / 4
    )

    # 5. Health score deviation
    df["health_dev"] = 1.0 - df["health_score"] / (df["health_score"].max() + 1)

    return df


BASE_FEATURES = [
    # Activity
    "steps", "calories_burned", "distance_covered", "exercise_duration",
    "ambient_temperature", "altitude", "uv_exposure",
    # Digital
    "screen_time", "notifications_received",
    # Vitals
    "heart_rate", "blood_oxygen_level", "skin_temperature",
    # Sleep
    "sleep_duration", "deep_sleep_duration", "rem_sleep_duration", "wakeups",
    # Body
    "body_fat_percentage", "muscle_mass", "bmi",
    # Nutrition
    "calories_intake", "water_intake", "health_score",
    # Encoded
    "stress_num", "mood_num", "ecg_num",
    "smoker_num", "medication_num", "snoring_num",
    "age_risk", "alcohol_num",
]

ENGINEERED = [
    "cardio_index", "sleep_index",
    "lifestyle_risk", "digital_stress", "health_dev",
]


def preprocess(df):
    print("\n[2/7] Preprocessing + feature engineering ...")
    df = engineer_features(df)
    y  = df["anomaly_flag"].values.astype(int)

    all_feats = BASE_FEATURES + ENGINEERED
    available = [c for c in all_feats if c in df.columns]
    X_raw = df[available].copy()
    X_raw = X_raw.fillna(X_raw.median(numeric_only=True))

    # StandardScaler for IF, SVM, RF, GB
    std_scaler = StandardScaler()
    X_std      = std_scaler.fit_transform(X_raw)

    # MinMaxScaler for Autoencoder (sigmoid output needs [0,1])
    mm_scaler  = MinMaxScaler()
    X_mm       = mm_scaler.fit_transform(X_raw)

    print(f"      Total features : {len(available)} "
          f"(base={len([c for c in BASE_FEATURES if c in df.columns])}, "
          f"engineered={len([c for c in ENGINEERED if c in df.columns])})")
    print(f"      Normal : {(y==0).sum()}  |  Anomaly : {(y==1).sum()}")
    print(f"      Anomaly rate : {(y==1).mean()*100:.1f}%")

    return X_std, X_mm, y, available, std_scaler, mm_scaler


# ╔══════════════════════════════════════════════════════════╗
# ║  3. PCA FOR AUTOENCODER                                 ║
# ╚══════════════════════════════════════════════════════════╝

def apply_pca(X_mm_tr, X_mm_te, variance=0.95):
    """
    Reduce dimensions while keeping 95% of variance.
    PCA removes correlated noise from the feature space,
    giving the Autoencoder a much cleaner signal to reconstruct.
    Normal vs anomaly becomes more separable in PCA space.
    """
    pca = PCA(n_components=variance, random_state=SEED)
    X_pca_tr = pca.fit_transform(X_mm_tr)
    X_pca_te = pca.transform(X_mm_te)
    print(f"\n      PCA: {X_mm_tr.shape[1]} features → "
          f"{X_pca_tr.shape[1]} components "
          f"(95% variance retained)")
    return X_pca_tr, X_pca_te, pca


# ╔══════════════════════════════════════════════════════════╗
# ║  3. SPLIT                                               ║
# ╚══════════════════════════════════════════════════════════╝

def split_data(X_std, X_mm, y):
    print("\n[3/7] Splitting data ...")
    (X_std_tr, X_std_te,
     X_mm_tr,  X_mm_te,
     y_tr,     y_te) = train_test_split(
        X_std, X_mm, y,
        test_size=0.25, random_state=SEED, stratify=y
    )
    print(f"      Train : {len(X_std_tr)}  "
          f"(normal={(y_tr==0).sum()}, anomaly={(y_tr==1).sum()})")
    print(f"      Test  : {len(X_std_te)}  "
          f"(normal={(y_te==0).sum()}, anomaly={(y_te==1).sum()})")
    return X_std_tr, X_std_te, X_mm_tr, X_mm_te, y_tr, y_te


# ╔══════════════════════════════════════════════════════════╗
# ║  HELPER                                                 ║
# ╚══════════════════════════════════════════════════════════╝

def _report(y_true, y_pred, tag):
    p   = precision_score(y_true, y_pred, zero_division=0)
    r   = recall_score(y_true,    y_pred, zero_division=0)
    mf1 = f1_score(y_true,        y_pred, average="macro", zero_division=0)
    try:    auc = roc_auc_score(y_true, y_pred)
    except: auc = float("nan")
    print(f"      [{tag:<20}]  P={p:.3f}  R={r:.3f}  "
          f"Macro-F1={mf1:.3f}  AUC={auc:.3f}")


def tune_threshold(scores, y_true):
    """
    Sweep 500 candidates. Maximise macro F1 on TRAINING set only.
    Macro F1 = average of Normal-F1 and Anomaly-F1.
    Prevents threshold collapsing to "flag everything as anomaly."
    """
    best_t, best_f1 = 0.0, 0.0
    for t in np.percentile(scores, np.linspace(1, 99, 500)):
        preds = (scores >= t).astype(int)
        mf1   = f1_score(y_true, preds, average="macro", zero_division=0)
        if mf1 > best_f1:
            best_f1, best_t = mf1, t
    return best_t, best_f1


# ╔══════════════════════════════════════════════════════════╗
# ║  4. ISOLATION FOREST                                    ║
# ╚══════════════════════════════════════════════════════════╝

def run_isolation_forest(X_std_tr, y_tr, X_std_te, y_te):
    print("\n[4/7] Isolation Forest ...")
    iso = IsolationForest(
        n_estimators=300, contamination=0.5,
        max_samples=min(1000, (y_tr==0).sum()),
        random_state=SEED, n_jobs=-1
    )
    iso.fit(X_std_tr[y_tr == 0])
    tr_scores = -iso.decision_function(X_std_tr)
    te_scores = -iso.decision_function(X_std_te)
    best_t, best_f1 = tune_threshold(tr_scores, y_tr)
    pred = (te_scores >= best_t).astype(int)
    print(f"      Threshold (train macro-F1={best_f1:.3f}): {best_t:.5f}")
    _report(y_te, pred, "Isolation Forest")
    return pred, te_scores


# ╔══════════════════════════════════════════════════════════╗
# ║  5. ONE-CLASS SVM                                       ║
# ╚══════════════════════════════════════════════════════════╝

def run_one_class_svm(X_std_tr, y_tr, X_std_te, y_te):
    """
    One-Class SVM learns a tight hypersphere boundary around
    normal samples in an RBF kernel space.

    Why better than Isolation Forest for this data:
    ─────────────────────────────────────────────────
    Your anomalies are defined by health PROFILE (chronic
    conditions, age, BMI), not just sensor spikes. The RBF
    kernel maps features into a high-dimensional space where
    the boundary between normal and anomaly is non-linear.
    Isolation Forest uses axis-aligned splits which can't
    capture this curved boundary as well.

    nu = upper bound on fraction of outliers in training data.
    We use the true normal fraction from training set.
    """
    print("\n[5/7] One-Class SVM ...")

    # Sub-sample for speed (SVM is O(n²))
    normal_idx = np.where(y_tr == 0)[0]
    n_fit = min(2000, len(normal_idx))
    rng   = np.random.default_rng(SEED)
    fit_idx = rng.choice(normal_idx, size=n_fit, replace=False)

    svm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
    svm.fit(X_std_tr[fit_idx])

    tr_scores = -svm.decision_function(X_std_tr)
    te_scores = -svm.decision_function(X_std_te)

    best_t, best_f1 = tune_threshold(tr_scores, y_tr)
    pred = (te_scores >= best_t).astype(int)
    print(f"      Threshold (train macro-F1={best_f1:.3f}): {best_t:.5f}")
    _report(y_te, pred, "One-Class SVM")
    return pred, te_scores


# ╔══════════════════════════════════════════════════════════╗
# ║  6. AUTOENCODER                                         ║
# ╚══════════════════════════════════════════════════════════╝

def build_autoencoder(n_in):
    """
    Architecture:  Input(n_in) → 64 → 32 → 16[latent]
                               → 32 → 64 → Output(n_in)

    Trained on PCA-reduced normal samples only.
    Uses MSE reconstruction error as anomaly score.
    """
    inp = layers.Input(shape=(n_in,), name="input")
    x   = layers.Dense(64, activation="relu")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.1)(x)
    x   = layers.Dense(32, activation="relu")(x)
    z   = layers.Dense(16, activation="relu", name="latent")(x)
    x   = layers.Dense(32, activation="relu")(z)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    out = layers.Dense(n_in, activation="sigmoid", name="output")(x)
    ae  = Model(inputs=inp, outputs=out, name="Autoencoder")
    ae.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss="mse")
    return ae


def train_autoencoder(X_pca_tr_norm, X_pca_val_norm, n_in):
    print("\n[6/7] Autoencoder (normal PCA-reduced samples only) ...")
    ae = build_autoencoder(n_in)
    ae.summary()
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=6, min_lr=1e-7, verbose=0),
    ]
    history = ae.fit(
        X_pca_tr_norm, X_pca_tr_norm,
        epochs=200, batch_size=64,
        validation_data=(X_pca_val_norm, X_pca_val_norm),
        callbacks=cb_list, verbose=1
    )
    return ae, history


def recon_error(ae, X):
    return np.mean((X - ae.predict(X, verbose=0)) ** 2, axis=1)


# ╔══════════════════════════════════════════════════════════╗
# ║  SUPERVISED MODELS                                      ║
# ╚══════════════════════════════════════════════════════════╝

def run_random_forest(X_tr, y_tr, X_te, y_te):
    print("\n      Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced_subsample",
        random_state=SEED, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    pred   = rf.predict(X_te)
    scores = rf.predict_proba(X_te)[:, 1]
    _report(y_te, pred, "Random Forest")
    return pred, scores, rf


def run_gradient_boosting(X_tr, y_tr, X_te, y_te):
    print("\n      Gradient Boosting ...")
    gb = HistGradientBoostingClassifier(
        max_iter=400, max_depth=8, learning_rate=0.05,
        min_samples_leaf=20, class_weight="balanced",
        random_state=SEED, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20, verbose=0
    )
    gb.fit(X_tr, y_tr)
    pred   = gb.predict(X_te)
    scores = gb.predict_proba(X_te)[:, 1]
    _report(y_te, pred, "Gradient Boosting")
    return pred, scores, gb


# ╔══════════════════════════════════════════════════════════╗
# ║  VISUALISATIONS                                         ║
# ╚══════════════════════════════════════════════════════════╝

C = dict(
    normal="#2ecc71", anomaly="#e74c3c", bg="#0f1117", card="#1a1d2e",
    text="#ecf0f1",   blue="#3498db",   gold="#f39c12", purple="#9b59b6",
    teal="#1abc9c",
)

def _style():
    plt.rcParams.update({
        "figure.facecolor": C["bg"],  "axes.facecolor": C["card"],
        "axes.edgecolor":  "#2c3e50", "axes.labelcolor": C["text"],
        "xtick.color": C["text"],     "ytick.color": C["text"],
        "text.color":  C["text"],     "grid.color":  "#2c3e50",
        "grid.linestyle": "--",       "grid.alpha":  0.5,
        "font.family": "monospace",
    })

def _save(fig, name):
    path = f"{OUTDIR}/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_training_loss(history):
    _style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"],     color=C["blue"], lw=2, label="Train")
    ax.plot(history.history["val_loss"], color=C["gold"], lw=2,
            linestyle="--", label="Validation")
    best_ep = np.argmin(history.history["val_loss"])
    ax.axvline(best_ep, color=C["normal"], lw=1.5, linestyle=":",
               label=f"Best epoch={best_ep}")
    ax.set_title("Autoencoder Training Loss (MSE)", fontsize=14)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.legend(framealpha=0.3); ax.grid(True)
    fig.tight_layout(); _save(fig, "01_training_loss.png")


def plot_error_vs_index(errors, y_te, threshold):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Scatter
    ax = axes[0]
    t = np.arange(len(errors))
    ax.scatter(t[y_te==0], errors[y_te==0], s=3, color=C["normal"],
               alpha=0.4, label="Normal")
    ax.scatter(t[y_te==1], errors[y_te==1], s=3, color=C["anomaly"],
               alpha=0.5, label="Anomaly (GT)")
    ax.axhline(threshold, color=C["gold"], lw=2, linestyle="--",
               label=f"Threshold={threshold:.5f}")
    ax.set_title("Reconstruction Error per Sample", fontsize=12)
    ax.set_xlabel("Sample Index"); ax.set_ylabel("MSE")
    ax.legend(framealpha=0.3); ax.grid(True)

    # Box plot comparison
    ax = axes[1]
    ax.boxplot([errors[y_te==0], errors[y_te==1]],
               labels=["Normal", "Anomaly"],
               patch_artist=True,
               boxprops=dict(facecolor=C["card"]),
               medianprops=dict(color=C["gold"], linewidth=2),
               whiskerprops=dict(color=C["text"]),
               capprops=dict(color=C["text"]),
               flierprops=dict(marker="o", color=C["anomaly"],
                               markersize=2, alpha=0.3))
    ax.axhline(threshold, color=C["gold"], lw=2, linestyle="--",
               label=f"Threshold={threshold:.5f}")
    ax.set_title("Error Distribution: Normal vs Anomaly", fontsize=12)
    ax.set_ylabel("MSE"); ax.legend(framealpha=0.3); ax.grid(True)

    fig.tight_layout(); _save(fig, "02_reconstruction_error.png")


def plot_feature_distributions(X_te, y_te, feature_names):
    _style()
    # Show the most clinically meaningful features
    key = ["cardio_index", "sleep_index", "lifestyle_risk",
           "health_dev", "heart_rate", "blood_oxygen_level"]
    indices = [feature_names.index(f) for f in key if f in feature_names][:6]
    if len(indices) < 6:
        extra = [i for i in range(len(feature_names)) if i not in indices]
        indices += extra[:6-len(indices)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    df_p = pd.DataFrame(X_te, columns=feature_names)
    df_p["label"] = ["Anomaly" if l==1 else "Normal" for l in y_te]
    for i, fi in enumerate(indices):
        fname = feature_names[fi]; ax = axes[i]
        for lbl, col in [("Normal",C["normal"]),("Anomaly",C["anomaly"])]:
            vals = df_p[df_p["label"]==lbl][fname]
            ax.hist(vals, bins=35, color=col, alpha=0.65,
                    label=lbl, density=True, edgecolor="none")
        ax.set_title(fname, fontsize=10)
        ax.set_xlabel("Scaled Value"); ax.set_ylabel("Density")
        ax.legend(fontsize=8, framealpha=0.3); ax.grid(True)
    fig.suptitle("Feature Distributions: Normal vs Anomaly",
                 fontsize=13, y=1.01)
    fig.tight_layout(); _save(fig, "03_feature_distributions.png")


def plot_confusion_matrices(y_te, pred_if, pred_svm, pred_ae,
                             pred_rf, pred_gb):
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    models = [
        (pred_if,  "Isolation Forest\n(unsupervised)"),
        (pred_svm, "One-Class SVM\n(unsupervised)"),
        (pred_ae,  "Autoencoder\n(unsupervised)"),
        (pred_rf,  "Random Forest\n(supervised)"),
        (pred_gb,  "Gradient Boosting\n(supervised)"),
    ]
    for ax, (preds, title) in zip(axes[:5], models):
        cm = confusion_matrix(y_te, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal","Anomaly"],
                    yticklabels=["Normal","Anomaly"],
                    linewidths=0.5, annot_kws={"size":13})
        p   = precision_score(y_te, preds, zero_division=0)
        r   = recall_score(y_te,    preds, zero_division=0)
        mf1 = f1_score(y_te,        preds, average="macro", zero_division=0)
        ax.set_title(f"{title}\nP={p:.3f}  R={r:.3f}  F1={mf1:.3f}",
                     fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    axes[5].axis("off")   # hide unused 6th panel
    fig.suptitle("Confusion Matrices — All Models", fontsize=14, y=1.01)
    fig.tight_layout(); _save(fig, "04_confusion_matrices.png")


def plot_score_histograms(errors_ae, scores_if, scores_svm,
                          threshold_ae, y_te):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, scores, thresh, title in [
        (axes[0], errors_ae,  threshold_ae, "Autoencoder"),
        (axes[1], scores_if,  None,          "Isolation Forest"),
        (axes[2], scores_svm, None,          "One-Class SVM"),
    ]:
        ax.hist(scores[y_te==0], bins=50, color=C["normal"],  alpha=0.7,
                label="Normal",  density=True, edgecolor="none")
        ax.hist(scores[y_te==1], bins=50, color=C["anomaly"], alpha=0.7,
                label="Anomaly", density=True, edgecolor="none")
        if thresh is not None:
            ax.axvline(thresh, color=C["gold"], lw=2, linestyle="--",
                       label=f"Threshold")
        ax.set_title(f"{title} Score Distribution", fontsize=11)
        ax.set_xlabel("Score"); ax.set_ylabel("Density")
        ax.legend(framealpha=0.3); ax.grid(True)
    fig.tight_layout(); _save(fig, "05_score_histograms.png")


def plot_model_comparison(y_te, pred_if, pred_svm, pred_ae,
                           pred_rf, pred_gb,
                           scores_if, scores_svm, errors_ae,
                           scores_rf, scores_gb):
    _style()
    metrics = ["Precision", "Recall", "Macro-F1", "ROC-AUC"]

    def get_vals(y, pred, score):
        try:    auc = roc_auc_score(y, score)
        except: auc = 0.0
        return [
            precision_score(y, pred, zero_division=0),
            recall_score(y,    pred, zero_division=0),
            f1_score(y,        pred, average="macro", zero_division=0),
            auc,
        ]

    all_models = [
        ("IF (unsupervised)",   pred_if,  scores_if,  C["gold"]),
        ("OCSVM (unsupervised)",pred_svm, scores_svm, C["teal"]),
        ("AE (unsupervised)",   pred_ae,  errors_ae,  C["blue"]),
        ("RF (supervised)",     pred_rf,  scores_rf,  C["normal"]),
        ("GB (supervised)",     pred_gb,  scores_gb,  C["purple"]),
    ]
    x = np.arange(len(metrics)); w = 0.14
    offsets = np.linspace(-2*w, 2*w, 5)
    fig, ax = plt.subplots(figsize=(15, 6))
    for (name, pred, score, col), offset in zip(all_models, offsets):
        vals = get_vals(y_te, pred, score)
        bars = ax.bar(x+offset, vals, w, label=name, color=col, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.008,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.3)
    ax.set_title("Model Comparison — All Five Models", fontsize=13)
    ax.set_ylabel("Score")
    ax.legend(framealpha=0.3, fontsize=9); ax.grid(True, axis="y")
    # Draw a dashed line at 0.75 as the target threshold
    ax.axhline(0.75, color=C["anomaly"], lw=1, linestyle=":",
               alpha=0.6, label="Target 0.75")
    fig.tight_layout(); _save(fig, "06_model_comparison.png")


def plot_roc_curves(y_te, scores_if, scores_svm, errors_ae,
                    scores_rf, scores_gb):
    _style()
    fig, ax = plt.subplots(figsize=(9, 7))
    for scores, label, col in [
        (scores_if,  "Isolation Forest (unsupervised)", C["gold"]),
        (scores_svm, "One-Class SVM (unsupervised)",    C["teal"]),
        (errors_ae,  "Autoencoder (unsupervised)",      C["blue"]),
        (scores_rf,  "Random Forest (supervised)",      C["normal"]),
        (scores_gb,  "Gradient Boosting (supervised)",  C["purple"]),
    ]:
        try:
            fpr, tpr, _ = roc_curve(y_te, scores)
            auc = roc_auc_score(y_te, scores)
            ax.plot(fpr, tpr, color=col, lw=2.5,
                    label=f"{label}  AUC={auc:.3f}")
        except Exception:
            pass
    ax.plot([0,1],[0,1], color="#7f8c8d", lw=1, linestyle="--",
            label="Random classifier (AUC=0.5)")
    ax.fill_between([0,1],[0,1], alpha=0.05, color="#7f8c8d")
    ax.set_title("ROC Curves — All Models", fontsize=14)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.legend(framealpha=0.3, fontsize=9, loc="lower right")
    ax.grid(True)
    fig.tight_layout(); _save(fig, "07_roc_curves.png")


def plot_health_dashboard(errors_ae, y_te, threshold,
                           pred_gb, scores_gb):
    _style()
    pred_ae = (errors_ae >= threshold).astype(int)
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Health Risk Dashboard — Wearable Sensor Monitor",
                 fontsize=16, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    # Timeline
    t = np.arange(len(errors_ae))
    ax1.scatter(t[y_te==0], errors_ae[y_te==0], s=3, color=C["normal"],
                alpha=0.4, label="Normal (GT)")
    ax1.scatter(t[y_te==1], errors_ae[y_te==1], s=3, color=C["anomaly"],
                alpha=0.6, label="Anomaly (GT)")
    ax1.axhline(threshold, color=C["gold"], lw=2.5, linestyle="--",
                label=f"Risk threshold = {threshold:.5f}")
    ax1.set_title("Autoencoder Reconstruction Error — All Test Samples",
                  fontsize=12)
    ax1.set_xlabel("Sample Index"); ax1.set_ylabel("MSE")
    ax1.legend(fontsize=10, framealpha=0.3); ax1.grid(True)

    # Ground truth pie
    n_n=(y_te==0).sum(); n_a=(y_te==1).sum()
    ax2.pie([n_n,n_a], labels=["Normal","Anomaly"],
            colors=[C["normal"],C["anomaly"]], autopct="%1.1f%%",
            startangle=90, wedgeprops={"edgecolor":C["bg"],"linewidth":2})
    ax2.set_title("Ground Truth Distribution", fontsize=11)

    # Risk severity tiers from AE
    p50,p75,p95 = np.percentile(errors_ae,[50,75,95])
    tiers  = ["✅ Low\n(<p50)","🟡 Mild\n(p50-75)",
               "🟠 Moderate\n(p75-95)","🔴 High\n(>p95)"]
    counts = [
        (errors_ae<p50).sum(),
        ((errors_ae>=p50)&(errors_ae<p75)).sum(),
        ((errors_ae>=p75)&(errors_ae<p95)).sum(),
        (errors_ae>=p95).sum(),
    ]
    bars = ax3.bar(tiers, counts,
                   color=[C["normal"],C["gold"],"#e67e22",C["anomaly"]],
                   edgecolor=C["bg"], linewidth=1.2)
    ax3.set_title("AE Risk Severity Tiers", fontsize=11)
    ax3.set_ylabel("Sample Count"); ax3.grid(True, axis="y")
    for bar, cnt in zip(bars, counts):
        ax3.text(bar.get_x()+bar.get_width()/2,
                 cnt+2, str(cnt),
                 ha="center", va="bottom", fontsize=9)

    # GB probability distribution
    ax4.hist(scores_gb[y_te==0], bins=40, color=C["normal"],
             alpha=0.7, label="Normal",  density=True, edgecolor="none")
    ax4.hist(scores_gb[y_te==1], bins=40, color=C["anomaly"],
             alpha=0.7, label="Anomaly", density=True, edgecolor="none")
    ax4.axvline(0.5, color=C["gold"], lw=2, linestyle="--",
                label="Decision boundary")
    ax4.set_title("Gradient Boosting — Risk Probability", fontsize=11)
    ax4.set_xlabel("P(Anomaly)"); ax4.set_ylabel("Density")
    ax4.legend(framealpha=0.3); ax4.grid(True)

    _save(fig, "08_health_risk_dashboard.png")


def plot_feature_importance(X_std, y, feature_names, rf_model):
    _style()
    df_tmp = pd.DataFrame(X_std, columns=feature_names)
    df_tmp["anomaly_flag"] = y
    corrs  = df_tmp.corr()["anomaly_flag"].drop("anomaly_flag").sort_values()
    rf_imp = pd.Series(rf_model.feature_importances_,
                       index=feature_names).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(feature_names)*0.32)))
    colors = [C["anomaly"] if v>0 else C["blue"] for v in corrs]
    axes[0].barh(corrs.index, corrs.values, color=colors, alpha=0.85)
    axes[0].axvline(0, color=C["text"], lw=0.8)
    axes[0].set_title("Pearson Correlation with Anomaly_Flag\n"
                       "Red=risk factor  Blue=protective", fontsize=11)
    axes[0].set_xlabel("Correlation"); axes[0].grid(True, axis="x")

    # Colour by engineered vs base
    eng_set = set(ENGINEERED)
    imp_colors = [C["gold"] if n in eng_set else C["purple"]
                  for n in rf_imp.index]
    axes[1].barh(rf_imp.index, rf_imp.values, color=imp_colors, alpha=0.85)
    axes[1].set_title("Random Forest Feature Importances\n"
                       "Gold=engineered  Purple=raw", fontsize=11)
    axes[1].set_xlabel("Importance"); axes[1].grid(True, axis="x")

    fig.tight_layout(); _save(fig, "09_feature_importance.png")


# ╔══════════════════════════════════════════════════════════╗
# ║  HEALTH INTERPRETATION                                  ║
# ╚══════════════════════════════════════════════════════════╝

def health_interpretation(errors_ae, y_te, threshold,
                           pred_ae, pred_if, pred_svm,
                           pred_rf, pred_gb):
    p75, p95 = np.percentile(errors_ae, [75, 95])
    mild     = ((errors_ae>threshold)&(errors_ae<=p75)).sum()
    moderate = ((errors_ae>p75)&(errors_ae<=p95)).sum()
    high     = (errors_ae>p95).sum()

    print("\n" + "="*60)
    print("  ❤  HEALTH RISK INTERPRETATION")
    print("="*60)
    print(f"""
  AE threshold  : {threshold:.5f}
  Normal        : {(errors_ae<=threshold).sum()} samples
  🟡 Mild risk  : {mild}
  🟠 Moderate   : {moderate}
  🔴 High risk  : {high}

  Composite Feature → Clinical Meaning:
  ──────────────────────────────────────
  cardio_index ↑    HR spike + SpO2 drop + stress + abnormal ECG
                    → Cardiovascular / arrhythmia risk
  sleep_index ↑     Wakeups + snoring - deep sleep - REM
                    → Sleep disorder / recovery deficit
  lifestyle_risk ↑  Low steps + smoking + alcohol + high BMI
                    → Metabolic syndrome / sedentary risk
  digital_stress ↑  High screen time + notifications + bad mood
                    → Circadian disruption / mental health risk
  health_dev ↑      Low health_score
                    → Composite wellbeing decline
    """)

    print("  Classification Report — Autoencoder:")
    print(classification_report(y_te, pred_ae,
          target_names=["Normal","Anomaly"], zero_division=0))

    print("  Classification Report — Gradient Boosting (best model):")
    print(classification_report(y_te, pred_gb,
          target_names=["Normal","Anomaly"], zero_division=0))


# ╔══════════════════════════════════════════════════════════╗
# ║  MAIN                                                   ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    # 1. Load
    df = load_and_merge()

    # 2. Preprocess
    X_std, X_mm, y, feature_names, std_scaler, mm_scaler = preprocess(df)

    # 3. Split
    (X_std_tr, X_std_te,
     X_mm_tr,  X_mm_te,
     y_tr,     y_te) = split_data(X_std, X_mm, y)

    # 4. Isolation Forest
    pred_if, scores_if = run_isolation_forest(X_std_tr, y_tr,
                                               X_std_te, y_te)

    # 5. One-Class SVM
    pred_svm, scores_svm = run_one_class_svm(X_std_tr, y_tr,
                                              X_std_te, y_te)

    # 6. Autoencoder on PCA-reduced data
    print("\n[6/7] Preparing PCA + Autoencoder ...")
    X_pca_tr, X_pca_te, pca = apply_pca(X_mm_tr, X_mm_te)

    # Normal-only split for AE training
    X_norm     = X_pca_tr[y_tr == 0]
    X_norm_tr, X_norm_val = train_test_split(
        X_norm, test_size=0.15, random_state=SEED
    )
    # Scale PCA output to [0,1] for sigmoid
    ae_scaler   = MinMaxScaler()
    X_norm_tr_s = ae_scaler.fit_transform(X_norm_tr)
    X_norm_val_s= ae_scaler.transform(X_norm_val)
    X_pca_tr_s  = ae_scaler.transform(X_pca_tr)
    X_pca_te_s  = ae_scaler.transform(X_pca_te)

    ae, history = train_autoencoder(X_norm_tr_s, X_norm_val_s,
                                     X_norm_tr_s.shape[1])
    errors_tr   = recon_error(ae, X_pca_tr_s)
    errors_te   = recon_error(ae, X_pca_te_s)
    threshold, tr_f1 = tune_threshold(errors_tr, y_tr)
    pred_ae = (errors_te >= threshold).astype(int)
    print(f"\n      AE threshold (train macro-F1={tr_f1:.3f}): {threshold:.5f}")
    _report(y_te, pred_ae, "Autoencoder")

    # 7. Supervised models
    print("\n[7/7] Supervised models ...")
    pred_rf, scores_rf, rf_model = run_random_forest(
        X_std_tr, y_tr, X_std_te, y_te)
    pred_gb, scores_gb, gb_model = run_gradient_boosting(
        X_std_tr, y_tr, X_std_te, y_te)

    # Visualisations
    print("\n      Generating visualisations ...")
    plot_training_loss(history)
    plot_error_vs_index(errors_te, y_te, threshold)
    plot_feature_distributions(X_std_te, y_te, feature_names)
    plot_confusion_matrices(y_te, pred_if, pred_svm, pred_ae,
                             pred_rf, pred_gb)
    plot_score_histograms(errors_te, scores_if, scores_svm,
                          threshold, y_te)
    plot_model_comparison(y_te, pred_if, pred_svm, pred_ae,
                           pred_rf, pred_gb,
                           scores_if, scores_svm, errors_te,
                           scores_rf, scores_gb)
    plot_roc_curves(y_te, scores_if, scores_svm, errors_te,
                    scores_rf, scores_gb)
    plot_health_dashboard(errors_te, y_te, threshold,
                           pred_gb, scores_gb)
    plot_feature_importance(X_std, y, feature_names, rf_model)

    health_interpretation(errors_te, y_te, threshold,
                           pred_ae, pred_if, pred_svm,
                           pred_rf, pred_gb)

    # Final summary
    print("\n" + "="*60)
    print("  FINAL MODEL SUMMARY")
    print("="*60)
    print(f"  {'Model':<26} {'Precision':>10} {'Recall':>8} "
          f"{'Macro-F1':>10} {'AUC':>8}")
    print("  " + "-"*64)
    for tag, pred, score in [
        ("Isolation Forest",  pred_if,  scores_if),
        ("One-Class SVM",     pred_svm, scores_svm),
        ("Autoencoder",       pred_ae,  errors_te),
        ("Random Forest",     pred_rf,  scores_rf),
        ("Gradient Boosting", pred_gb,  scores_gb),
    ]:
        p   = precision_score(y_te, pred, zero_division=0)
        r   = recall_score(y_te,    pred, zero_division=0)
        mf1 = f1_score(y_te,        pred, average="macro", zero_division=0)
        try:    auc = roc_auc_score(y_te, score)
        except: auc = float("nan")
        print(f"  {tag:<26} {p:>10.3f} {r:>8.3f} {mf1:>10.3f} {auc:>8.3f}")

    print(f"\n  ✅  Pipeline complete! 9 plots → ./{OUTDIR}/")
    print("="*60)


if __name__ == "__main__":
    main()