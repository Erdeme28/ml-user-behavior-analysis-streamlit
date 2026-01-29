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

st.set_page_config(layout="wide")

st.title("🔍 Classification Playground")
st.markdown(
    """
    This section allows interactive execution of classification models
    to predict users who will become premium subscribers.
    """
)

st.sidebar.header("⚙️ Model Configuration")

model_name = st.sidebar.selectbox(
    "Choose algorithm",
    ("Logistic Regression", "KNN", "Decision Tree", "Random Forest")
)

sampling_method = st.sidebar.selectbox(
    "Data balancing method",
    ("None", "SMOTE", "NearMiss")
)

n_samples = st.sidebar.slider(
    "Number of samples",
    min_value=2000,
    max_value=12000,
    value=8000,
    step=1000
)

run_button = st.sidebar.button("🚀 Run classification")

base_df, delta_df = generate_dataset(n_samples=n_samples)

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
    st.subheader("📊 Classification Results")

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, stratify=labels, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    if model_name == "KNN":
        X_train = scaler.fit_transform(X_train)
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
        model = RandomForestClassifier(n_estimators=300, random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("F1-score", f"{f1:.3f}")

    st.markdown("### 🧩 Confusion Matrix")
    matrix = confusion_matrix(y_test, predictions)

    matrix_df = pd.DataFrame(
        matrix,
        index=["Not Premium", "Premium"],
        columns=["Predicted No", "Predicted Yes"]
    )

    fig, ax = plt.subplots()
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

else:
    st.info(
        "Select the model and preprocessing options, then click "
        "**Run classification** to evaluate performance metrics."
    )
