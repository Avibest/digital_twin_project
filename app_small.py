import os
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from pathlib import Path
import zipfile
import os

# --- Ensure data_small folder exists on the server ---
DATA_DIR = Path("data_small")
ZIP_PATH = Path("data_small.zip")

if ZIP_PATH.exists() and not DATA_DIR.exists():
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(".")   # this will recreate the data_small/ folder

# ------------------------------
#  CONFIG
# ------------------------------
DATA_DIR = "data_small"   # expects data/1st_test, data/2nd_test, data/3rd_test


# ------------------------------
#  DATA LOADING
# ------------------------------
def load_test_set(test_name):
    """
    Loads all data files from a given test folder.

    IMS bearing files often have NO extension (e.g., '2003.10.22.12.06.24'),
    so we:
      - walk through all subfolders
      - skip obvious non-data files (.pdf, .rar, .zip)
      - load everything else with np.loadtxt
    """
    folder_path = os.path.join(DATA_DIR, test_name)

    data_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith((".pdf", ".rar", ".zip")):
                continue
            data_files.append(os.path.join(root, f))

    data_files = sorted(data_files)
    data = [np.loadtxt(path) for path in data_files]

    return data, data_files


def compute_more_features(test_name, channel=0):
    """
    For every file in a test, compute:
      - RMS
      - kurtosis
      - crest factor
      - spectral energy
    Returns a dict of numpy arrays.
    """
    data, files = load_test_set(test_name)

    rms_list = []
    kurtosis_list = []
    crest_factor_list = []
    spectral_energy_list = []

    for sig in data:
        x = sig[:, channel] if sig.ndim == 2 else sig

        # RMS
        rms = np.sqrt(np.mean(x**2))

        # Kurtosis (spikiness)
        std = np.std(x)
        if std == 0:
            kurt = 0.0
        else:
            kurt = np.mean((x - np.mean(x))**4) / (std**4)

        # Crest factor (peak/RMS)
        crest = np.max(np.abs(x)) / (rms + 1e-12)

        # Spectral energy via Welch PSD
        freqs, psd = welch(x, fs=20000)  # 20 kHz assumed sampling
        spec_energy = np.sum(psd)

        rms_list.append(rms)
        kurtosis_list.append(kurt)
        crest_factor_list.append(crest)
        spectral_energy_list.append(spec_energy)

    return {
        "rms": np.array(rms_list),
        "kurtosis": np.array(kurtosis_list),
        "crest_factor": np.array(crest_factor_list),
        "spectral_energy": np.array(spectral_energy_list),
        "files": files,
    }


# ------------------------------
#  MODEL TRAINING
# ------------------------------
@st.cache_resource
def train_model():
    """
    Loads 1st_test (healthy) and 3rd_test (failing),
    extracts features, trains a Random Forest classifier,
    and returns the trained model plus feature stats.
    """
    feats_1 = compute_more_features("1st_test", channel=0)  # healthy-ish
    feats_3 = compute_more_features("3rd_test", channel=0)  # failure run

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

    # Train/test split (for info only)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    # We'll also keep mean+std to optionally normalize later if we want
    feature_stats = {
        "mean": X.mean(axis=0),
        "std": X.std(axis=0) + 1e-12,
    }

    return clf, feature_stats, (train_acc, test_acc)


# ------------------------------
#  FEATURE EXTRACTION FOR ONE SIGNAL
# ------------------------------
def compute_features_for_signal(x):
    """
    Compute the same 4 features for a single 1D signal x.
    Returns a 1D numpy array [rms, kurtosis, crest_factor, spectral_energy].
    """
    x = np.asarray(x)
    rms = np.sqrt(np.mean(x**2))

    std = np.std(x)
    if std == 0:
        kurt = 0.0
    else:
        kurt = np.mean((x - np.mean(x))**4) / (std**4)

    crest = np.max(np.abs(x)) / (rms + 1e-12)

    freqs, psd = welch(x, fs=20000)
    spec_energy = np.sum(psd)

    return np.array([rms, kurt, crest, spec_energy])


# ------------------------------
#  STREAMLIT APP
# ------------------------------
def main():
    st.title("Bearing Health Explorer – NASA IMS Data")
    st.write(
        "This app uses vibration data from the NASA IMS bearing dataset, "
        "extracts health features, and predicts whether a bearing is "
        "in a healthy or failing state."
    )

    # Train/load model
    with st.spinner("Training model (only runs once)..."):
        clf, feature_stats, (train_acc, test_acc) = train_model()

    st.sidebar.header("Controls")

    test_choice = st.sidebar.selectbox(
        "Select test run",
        ["1st_test (mostly healthy)", "3rd_test (failure run)"],
    )

    if "1st_test" in test_choice:
        test_name = "1st_test"
        label_hint = "This run is mostly healthy."
    else:
        test_name = "3rd_test"
        label_hint = "This run contains a bearing failure."

    st.sidebar.write(label_hint)

    # Load data for selected test
    data, files = load_test_set(test_name)
    num_signals = len(data)

    idx = st.sidebar.slider(
        "Measurement index",
        min_value=0,
        max_value=max(num_signals - 1, 0),
        value=0,
        step=1,
    )

    st.write(f"Selected file: `{os.path.basename(files[idx])}`")

    # Plot the raw signal
    x = data[idx]
    x = x[:, 0] if x.ndim == 2 else x

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x)
    ax.set_title(f"Vibration signal – {test_name}, sample {idx}")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    st.pyplot(fig)

    # Predict health
    feats = compute_features_for_signal(x).reshape(1, -1)
    prob_failing = clf.predict_proba(feats)[0, 1]
    pred_label = clf.predict(feats)[0]

    st.subheader("Model prediction")

    if pred_label == 0:
        st.success(f"Predicted state: **HEALTHY**  (failure probability: {prob_failing*100:.2f}%)")
    else:
        st.error(f"Predicted state: **FAILING**  (failure probability: {prob_failing*100:.2f}%)")

    st.markdown("---")
    st.write(f"**Model train accuracy:** {train_acc*100:.2f}%")
    st.write(f"**Model test accuracy:** {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()

