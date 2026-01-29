import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

from src.data.data_generator import generate_dataset
from src.config import OUT_DIR, OUT_CM_DIR

st.set_page_config(layout="wide")

st.title("💾 Export Results")
st.markdown(
    """
    This section allows saving the results obtained from the interface
    (metrics and confusion matrix) directly into the `outputs/` folder.
    """
)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_CM_DIR, exist_ok=True)

st.sidebar.header("⚙️ Export Configuration")

model_name = st.sidebar.selectbox(
    "Classification model",
    ("Logistic Regression", "KNN", "Decision Tree", "Random Forest")
)

sampling_method = st.sidebar.selectbox(
    "Sampling method",
    ("None", "SMOTE", "NearMiss")
)

n_samples = st.sidebar.slider("Number of samples", 2000, 12000, 8000, 1000)

experiment_name = st.sidebar.text_input(
    "Experiment name (file prefix)",
    value="experiment_1"
)

run_button = st.sidebar.button("🚀 Run and export")

base_df, delta_df = generate_dataset(n_samples=int(n_samples))

features = delta_df[
    [
        "delta_friend_cnt",
        "delta_songs",
        "delta_loved",
        "delta_posts",
        "delta_playlists",
        "delta_shouts",
    ]
].values

labels = base_df["adopter"].values

if run_button:
    st.subheader("📊 Results and Export")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        stratify=labels,
        test_size=0.2,
        random_state=42,
    )

    if model_name == "KNN":
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    if sampling_method == "SMOTE":
        sampler = SMOTE(random_state=42)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    elif sampling_method == "NearMiss":
        sampler = NearMiss()
        X_train, y_train = sampler.fit_resample(X_train, y_train)

    if model_name == "Logistic Regression":
        model = LogisticRegression(solver="liblinear")
    elif model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5)
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    st.metric("Accuracy", f"{accuracy:.3f}")
    st.metric("F1-score", f"{f1:.3f}")

    matrix = confusion_matrix(y_test, predictions)
    matrix_df = pd.DataFrame(
        matrix,
        index=["Not Premium", "Premium"],
        columns=["Predicted No", "Predicted Yes"],
    )

    fig, ax = plt.subplots()
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} | {sampling_method}")
    st.pyplot(fig)

    confusion_matrix_path = os.path.join(
        OUT_CM_DIR,
        f"{experiment_name}_cm.png",
    )
    fig.savefig(confusion_matrix_path)
    plt.close(fig)

    metrics_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "sampling": sampling_method,
                "n_samples": n_samples,
                "accuracy": accuracy,
                "f1": f1,
            }
        ]
    )

    metrics_path = os.path.join(
        OUT_DIR,
        f"{experiment_name}_metrics.csv",
    )
    metrics_df.to_csv(metrics_path, index=False)

    st.success("✅ Results were successfully saved.")
    st.code(f"{confusion_matrix_path}\n{metrics_path}")

else:
    st.info(
        "Configure the model and preprocessing options, then click "
        "**Run and export** to save the results."
    )
