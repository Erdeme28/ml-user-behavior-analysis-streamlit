import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

from src.data.data_generator import generate_dataset

st.set_page_config(layout="wide")

st.title("🏁 Model Comparison / Leaderboard")
st.markdown(
    """
    This page runs multiple models and builds a comparison table.
    It is useful for justifying the final model choice and
    experimentally demonstrating that a certain algorithm
    performs better.
    """
)

st.sidebar.header("⚙️ Benchmark Configuration")

n_samples = st.sidebar.slider("Number of samples", 2000, 12000, 8000, 1000)
test_size = st.sidebar.slider("Test size (%)", 10, 40, 20, 5) / 100.0
random_state = st.sidebar.number_input("Random seed", 0, 10_000, 42, 1)

run_button = st.sidebar.button("🚀 Run Leaderboard")

st.sidebar.markdown("---")
st.sidebar.caption("Note: KNN uses scaling. Sampling is applied only on training data.")

base_df, delta_df = generate_dataset(
    n_samples=int(n_samples),
    random_state=int(random_state),
)

X_classification = delta_df[
    [
        "delta_friend_cnt",
        "delta_songs",
        "delta_loved",
        "delta_posts",
        "delta_playlists",
        "delta_shouts",
    ]
].values
y_classification = base_df["adopter"].values

X_regression = base_df[
    ["songsListened", "lovedTracks", "posts", "playlists", "shouts", "tenure"]
].values
y_regression = delta_df["engagement"].values


def evaluate_classification_models(X, y, test_size, seed) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=seed
    )

    results = []

    def add_result(model_name, y_true, y_pred):
        results.append(
            {
                "task": "classification",
                "model": model_name,
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred),
            }
        )

    model = LogisticRegression(solver="liblinear")
    model.fit(X_train, y_train)
    add_result("Logistic Regression (baseline)", y_test, model.predict(X_test))

    sampler = SMOTE(random_state=seed)
    X_sm, y_sm = sampler.fit_resample(X_train, y_train)
    model_sm = LogisticRegression(solver="liblinear")
    model_sm.fit(X_sm, y_sm)
    add_result("Logistic Regression + SMOTE", y_test, model_sm.predict(X_test))

    sampler_nm = NearMiss()
    X_nm, y_nm = sampler_nm.fit_resample(X_train, y_train)
    model_nm = LogisticRegression(solver="liblinear")
    model_nm.fit(X_nm, y_nm)
    add_result("Logistic Regression + NearMiss", y_test, model_nm.predict(X_test))

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    add_result("KNN (scaled baseline)", y_test, knn.predict(X_test_scaled))

    sampler_knn = SMOTE(random_state=seed)
    X_knn_sm, y_knn_sm = sampler_knn.fit_resample(X_train_scaled, y_train)
    knn_sm = KNeighborsClassifier(n_neighbors=5)
    knn_sm.fit(X_knn_sm, y_knn_sm)
    add_result("KNN (scaled) + SMOTE", y_test, knn_sm.predict(X_test_scaled))

    tree = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=5,
        random_state=seed,
    )
    tree.fit(X_train, y_train)
    add_result("Decision Tree", y_test, tree.predict(X_test))

    forest = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
    )
    forest.fit(X_train, y_train)
    add_result("Random Forest", y_test, forest.predict(X_test))

    return pd.DataFrame(results)


def evaluate_regression_models(X, y, test_size, seed) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    results = []

    def add_result(model_name, y_true, y_pred):
        results.append(
            {
                "task": "regression",
                "model": model_name,
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": r2_score(y_true, y_pred),
            }
        )

    linear = LinearRegression()
    linear.fit(X_train, y_train)
    add_result("Linear Regression", y_test, linear.predict(X_test))

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsRegressor(n_neighbors=30)
    knn.fit(X_train_scaled, y_train)
    add_result("KNN Regressor (scaled)", y_test, knn.predict(X_test_scaled))

    forest = RandomForestRegressor(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
    )
    forest.fit(X_train, y_train)
    add_result("Random Forest Regressor", y_test, forest.predict(X_test))

    return pd.DataFrame(results)


if run_button:
    st.subheader("📌 Running benchmark...")

    classification_results = evaluate_classification_models(
        X_classification,
        y_classification,
        test_size,
        int(random_state),
    )

    regression_results = evaluate_regression_models(
        X_regression,
        y_regression,
        test_size,
        int(random_state),
    )

    st.markdown("## 🧠 Classification Leaderboard")
    classification_sorted = classification_results.sort_values(
        by="f1",
        ascending=False,
    ).reset_index(drop=True)
    st.dataframe(classification_sorted, use_container_width=True)

    best_classification = classification_sorted.iloc[0]
    st.success(
        f"🏆 Best classification model (by F1): "
        f"**{best_classification['model']}** | "
        f"F1={best_classification['f1']:.3f}"
    )

    st.markdown("## 📈 Regression Leaderboard")
    regression_sorted = regression_results.sort_values(
        by="r2",
        ascending=False,
    ).reset_index(drop=True)
    st.dataframe(regression_sorted, use_container_width=True)

    best_regression = regression_sorted.iloc[0]
    st.success(
        f"🏆 Best regression model (by R²): "
        f"**{best_regression['model']}** | "
        f"R²={best_regression['r2']:.4f}"
    )

    st.markdown("---")
    st.caption(
        "Note: you can change the number of samples and the random seed "
        "to observe result stability."
    )


else:
    st.info(
        "Run the benchmark to compare multiple models "
        "using consistent train/test splits and evaluation metrics."
    )
