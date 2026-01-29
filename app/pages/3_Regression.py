import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.data_generator import generate_dataset

st.set_page_config(layout="wide")

st.title("📈 Regression Playground")
st.markdown(
    """
    In this section we estimate the engagement score (a continuous variable)
    using regression models. You can compare multiple algorithms
    and visualize the Real vs Predicted relationship.
    """
)

st.sidebar.header("⚙️ Regression Configuration")

model_name = st.sidebar.selectbox(
    "Choose regression model",
    ("Linear Regression", "KNN Regressor", "Random Forest Regressor")
)

n_samples = st.sidebar.slider(
    "Number of samples",
    min_value=2000,
    max_value=12000,
    value=8000,
    step=1000
)

test_size = st.sidebar.slider(
    "Test size (%)",
    min_value=10,
    max_value=40,
    value=20,
    step=5
) / 100.0

random_state = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=10_000,
    value=42,
    step=1
)

st.sidebar.subheader("🔧 KNN (only if selected)")
knn_neighbors = st.sidebar.slider("n_neighbors (K)", 3, 75, 30, 1)

st.sidebar.subheader("🌲 Random Forest (only if selected)")
rf_estimators = st.sidebar.slider("n_estimators", 50, 600, 300, 50)
rf_max_depth_option = st.sidebar.selectbox("max_depth", ("None", "10", "20", "30"))

use_scaling = st.sidebar.checkbox(
    "Use feature scaling (recommended for KNN)",
    value=True
)

run_button = st.sidebar.button("🚀 Run regression")

base_df, delta_df = generate_dataset(
    n_samples=n_samples,
    random_state=int(random_state)
)

features = base_df[
    ["songsListened", "lovedTracks", "posts", "playlists", "shouts", "tenure"]
].values

target = delta_df["engagement"].values

with st.expander("🔎 Data preview (first 5 rows)"):
    preview_df = base_df.join(delta_df[["engagement"]]).head(5)
    st.dataframe(preview_df, use_container_width=True)

if run_button:
    st.subheader("📊 Regression Results")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=int(random_state)
    )

    if use_scaling:
        scaler = StandardScaler().fit(X_train)
        X_train_used = scaler.transform(X_train)
        X_test_used = scaler.transform(X_test)
    else:
        X_train_used = X_train
        X_test_used = X_test

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "KNN Regressor":
        model = KNeighborsRegressor(n_neighbors=int(knn_neighbors))
    else:
        if rf_max_depth_option == "None":
            max_depth = None
        else:
            max_depth = int(rf_max_depth_option)

        model = RandomForestRegressor(
            n_estimators=int(rf_estimators),
            random_state=int(random_state),
            n_jobs=-1,
            max_depth=max_depth
        )

    model.fit(X_train_used, y_train)
    predictions = model.predict(X_test_used)

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.2f}")
    c2.metric("RMSE", f"{rmse:.2f}")
    c3.metric("R²", f"{r2:.4f}")

    st.markdown("### 📌 Real vs Predicted")

    plot_size = min(1200, len(y_test))
    indices = np.random.choice(len(y_test), size=plot_size, replace=False)

    fig = plt.figure(figsize=(7, 5))
    plt.scatter(y_test[indices], predictions[indices], alpha=0.5)

    min_value = float(min(y_test[indices].min(), predictions[indices].min()))
    max_value = float(max(y_test[indices].max(), predictions[indices].max()))

    plt.plot([min_value, max_value], [min_value, max_value])
    plt.xlabel("True engagement")
    plt.ylabel("Predicted engagement")
    plt.title(f"{model_name} | Real vs Predicted")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### 🧪 Residuals (True - Predicted)")
    residuals = y_test - predictions

    fig2 = plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins=40)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("### 🧾 Results sample")
    results_sample = pd.DataFrame({
        "true_engagement": y_test[:15],
        "predicted_engagement": predictions[:15],
        "residual": residuals[:15]
    })
    st.dataframe(results_sample, use_container_width=True)

else:
    st.info(
        "Select the model and preprocessing options, then click "
        "**Run regression** to evaluate performance metrics."
    )
