import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.data.data_generator import generate_dataset

st.set_page_config(layout="wide")

st.title("🧠 Explainability (Feature Importance)")
st.markdown(
    """
    This section interprets tree-based models (Decision Tree / Random Forest)
    using feature importance.
    The goal is to understand which variables influence the prediction the most.
    """
)

st.sidebar.header("⚙️ Explainability Configuration")

model_name = st.sidebar.selectbox(
    "Interpretable model",
    ("Decision Tree", "Random Forest")
)

n_samples = st.sidebar.slider("Number of samples", 2000, 12000, 8000, 1000)
test_size = st.sidebar.slider("Test size (%)", 10, 40, 20, 5) / 100.0
random_state = st.sidebar.number_input("Random seed", 0, 10_000, 42, 1)

st.sidebar.subheader("🌳 Decision Tree parameters")
dt_max_depth_option = st.sidebar.selectbox(
    "max_depth",
    ("None", "6", "10", "15")
)

st.sidebar.subheader("🌲 Random Forest parameters")
rf_estimators = st.sidebar.slider("n_estimators", 50, 600, 300, 50)

run_button = st.sidebar.button("🚀 Train and explain")

base_df, delta_df = generate_dataset(
    n_samples=int(n_samples),
    random_state=int(random_state),
)

feature_names = [
    "delta_friend_cnt",
    "delta_songs",
    "delta_loved",
    "delta_posts",
    "delta_playlists",
    "delta_shouts",
]

features = delta_df[feature_names].values
labels = base_df["adopter"].values

with st.expander("🔎 Feature preview"):
    st.dataframe(
        delta_df[feature_names].head(10),
        use_container_width=True,
    )

if run_button:
    st.subheader("📊 Model performance")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        stratify=labels,
        test_size=float(test_size),
        random_state=int(random_state),
    )

    if model_name == "Decision Tree":
        max_depth = None if dt_max_depth_option == "None" else int(dt_max_depth_option)
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=5,
            random_state=int(random_state),
        )
    else:
        model = RandomForestClassifier(
            n_estimators=int(rf_estimators),
            random_state=int(random_state),
            n_jobs=-1,
        )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{accuracy:.3f}")
    c2.metric("F1-score", f"{f1:.3f}")

    with st.expander("🧩 Confusion matrix (raw values)"):
        st.write(confusion_matrix(y_test, predictions))

    st.markdown("---")
    st.subheader("⭐ Feature importances")

    importances = model.feature_importances_
    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(importance_df, use_container_width=True)

    fig = plt.figure(figsize=(8, 4))
    plt.bar(importance_df["feature"], importance_df["importance"])
    plt.xticks(rotation=30, ha="right")
    plt.title(f"{model_name} - Feature importance")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### 🧠 Interpretation")
    top_feature = importance_df.iloc[0]["feature"]
    top_value = importance_df.iloc[0]["importance"]

    st.write(
        f"- **Most influential feature**: `**{top_feature}**` "
        f"(importance ≈ {top_value:.3f}).\n"
        f"- Interpretation: the model relies on this variable the most "
        f"to separate users who become premium from those who do not.\n"
        f"- In this dataset, activity-related features "
        f"(songs, loved tracks, posts, playlists, shouts) "
        f"tend to have the highest impact."
        f"- Note: feature importance reflects model behavior on synthetic data and does not imply causal relationships."
    )

else:
    st.info(
        "Select the model and parameters, then click "
        "**Train and explain** to analyze feature importance."
    )
