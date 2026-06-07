import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
    auc,
)

# ------------------------------
# CONFIG
# ------------------------------
DATA_DIR = Path("data")         # assumes full IMS data is here: data/1st_test, data/3rd_test
FIG_DIR = Path("figures")       # output folder for all PDFs
FIG_DIR.mkdir(exist_ok=True)


# ------------------------------
# DATA LOADING AND FEATURES
# ------------------------------
def load_test_set(test_name, max_files=None):
    """
    Load all (or first max_files) data files from a given test folder.
    Skips obvious non-data extensions.
    """
    folder_path = DATA_DIR / test_name
    data_files = []

    for root, dirs, files in os.walk(folder_path):
        for f in files:
            # Skip archives, docs etc
            lf = f.lower()
            if lf.endswith((".pdf", ".rar", ".zip", ".doc", ".docx")):
                continue
            data_files.append(Path(root) / f)

    data_files = sorted(data_files)
    if max_files is not None:
        data_files = data_files[:max_files]

    data = [np.loadtxt(path) for path in data_files]
    return data, data_files


def compute_features_for_test(test_name, channel=0, max_files=None):
    """
    Compute RMS, kurtosis, crest factor and spectral energy
    for every file in a given test.
    Returns a dict of numpy arrays.
    """
    data, files = load_test_set(test_name, max_files=max_files)

    rms_list = []
    kurtosis_list = []
    crest_list = []
    spec_energy_list = []

    for sig in data:
        x = sig[:, channel] if sig.ndim == 2 else sig

        # RMS
        rms = np.sqrt(np.mean(x**2))

        # Kurtosis
        std = np.std(x)
        if std == 0:
            kurt = 0.0
        else:
            kurt = np.mean((x - np.mean(x)) ** 4) / (std ** 4)

        # Crest factor
        crest = np.max(np.abs(x)) / (rms + 1e-12)

        # Spectral energy via Welch PSD
        fs = 20000  # Hz, IMS default sampling
        freqs, psd = welch(x, fs=fs)
        spec_energy = np.sum(psd)

        rms_list.append(rms)
        kurtosis_list.append(kurt)
        crest_list.append(crest)
        spec_energy_list.append(spec_energy)

    return {
        "rms": np.array(rms_list),
        "kurtosis": np.array(kurtosis_list),
        "crest_factor": np.array(crest_list),
        "spectral_energy": np.array(spec_energy_list),
        "files": files,
    }


# ------------------------------
# MODEL TRAINING
# ------------------------------
def train_classifier(feats_1, feats_3):
    """
    Build feature matrix X and labels y from 1st and 3rd tests,
    then train a Random Forest classifier.
    """
    X_healthy = np.column_stack([
        feats_1["rms"],
        feats_1["kurtosis"],
        feats_1["crest_factor"],
        feats_1["spectral_energy"],
    ])

    X_failing = np.column_stack([
        feats_3["rms"],
        feats_3["kurtosis"],
        feats_3["crest_factor"],
        feats_3["spectral_energy"],
    ])

    y_healthy = np.zeros(len(X_healthy), dtype=int)
    y_failing = np.ones(len(X_failing), dtype=int)

    X = np.vstack([X_healthy, X_failing])
    y = np.concatenate([y_healthy, y_failing])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return clf, X_train, X_test, y_train, y_test, y_pred, y_prob


# ------------------------------
# PLOTTING HELPERS
# ------------------------------
def save_fig(name):
    """
    Save the current matplotlib figure as a PDF in the figures folder.
    """
    out_path = FIG_DIR / f"{name}.pdf"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ------------------------------
# MAIN FIGURE PIPELINE
# ------------------------------
def main():
    # You can adjust max_files if full dataset is too big
    # None means "use all files available"
    feats_1 = compute_features_for_test("1st_test", channel=0, max_files=None)
    feats_3 = compute_features_for_test("3rd_test", channel=0, max_files=None)

    # 1) RMS trend
    plt.figure(figsize=(8, 4))
    plt.plot(feats_1["rms"], label="1st test (healthy)")
    plt.plot(feats_3["rms"], label="3rd test (failing)")
    plt.title("RMS Trend Over Time")
    plt.xlabel("Sample index")
    plt.ylabel("RMS value")
    plt.legend()
    plt.grid(True)
    save_fig("rms_trend")

    # 2) Kurtosis trend
    plt.figure(figsize=(8, 4))
    plt.plot(feats_1["kurtosis"], label="1st test (healthy)")
    plt.plot(feats_3["kurtosis"], label="3rd test (failing)")
    plt.title("Kurtosis Trend Over Time")
    plt.xlabel("Sample index")
    plt.ylabel("Kurtosis")
    plt.legend()
    plt.grid(True)
    save_fig("kurtosis_trend")

    # 3) Spectral energy trend
    plt.figure(figsize=(8, 4))
    plt.plot(feats_1["spectral_energy"], label="1st test (healthy)")
    plt.plot(feats_3["spectral_energy"], label="3rd test (failing)")
    plt.title("Spectral Energy Trend Over Time")
    plt.xlabel("Sample index")
    plt.ylabel("Spectral energy")
    plt.legend()
    plt.grid(True)
    save_fig("spectral_energy_trend")

    # 4) Health score over time (3rd test only)
    rms_3 = feats_3["rms"]
    kurt_3 = feats_3["kurtosis"]
    crest_3 = feats_3["crest_factor"]
    spec_3 = feats_3["spectral_energy"]

    # avoid division by zero
    health_score = (
        rms_3 / (np.max(rms_3) + 1e-12)
        + kurt_3 / (np.max(kurt_3) + 1e-12)
        + crest_3 / (np.max(crest_3) + 1e-12)
        + spec_3 / (np.max(spec_3) + 1e-12)
    )

    plt.figure(figsize=(8, 4))
    plt.plot(health_score)
    plt.title("Combined Health Score Over Time (3rd Test)")
    plt.xlabel("Sample index")
    plt.ylabel("Normalized health score")
    plt.grid(True)
    save_fig("health_score_3rd_test")

    # 5) Train classifier and compute metrics
    clf, X_train, X_test, y_train, y_test, y_pred, y_prob = train_classifier(
        feats_1, feats_3
    )

    # 6) Feature importance
    feature_names = ["RMS", "Kurtosis", "Crest factor", "Spectral energy"]
    importances = clf.feature_importances_

    plt.figure(figsize=(6, 4))
    plt.bar(feature_names, importances)
    plt.title("Feature Importance – Random Forest")
    plt.ylabel("Importance")
    save_fig("feature_importance")

    # 7) Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Failing"])
    disp.plot()
    plt.title("Confusion Matrix – Bearing Classification")
    save_fig("confusion_matrix")

    # 8) ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve – Bearing Failure Detection")
    plt.legend()
    plt.grid(True)
    save_fig("roc_curve")


if __name__ == "__main__":
    main()
