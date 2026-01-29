from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from src.evaluation.metrics import (
    compute_regression_metrics,
    RegressionMetrics,
)


def run_regression(base_df, delta_df) -> list[RegressionMetrics]:
    """
    Runs regression experiments for the engagement target.
    Focus: model comparison and scaling for KNN.
    """
    features = base_df[
        ["songsListened", "lovedTracks", "posts", "playlists", "shouts", "tenure"]
    ].values

    target = delta_df["engagement"].values

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    results: list[RegressionMetrics] = []

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = compute_regression_metrics(
        "Linear Regression",
        y_test,
        predictions,
    )
    results.append(metrics)
    print(
        f"{metrics.name}: "
        f"MAE={metrics.mae:.4f}, "
        f"RMSE={metrics.rmse:.4f}, "
        f"R2={metrics.r2:.4f}",
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_knn = KNeighborsRegressor(n_neighbors=30)
    model_knn.fit(X_train_scaled, y_train)
    predictions = model_knn.predict(X_test_scaled)

    metrics = compute_regression_metrics(
        "KNN Regressor (scaled)",
        y_test,
        predictions,
    )
    results.append(metrics)
    print(
        f"{metrics.name}: "
        f"MAE={metrics.mae:.4f}, "
        f"RMSE={metrics.rmse:.4f}, "
        f"R2={metrics.r2:.4f}",
    )

    model_forest = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model_forest.fit(X_train, y_train)
    predictions = model_forest.predict(X_test)

    metrics = compute_regression_metrics(
        "Random Forest Regressor",
        y_test,
        predictions,
    )
    results.append(metrics)
    print(
        f"{metrics.name}: "
        f"MAE={metrics.mae:.4f}, "
        f"RMSE={metrics.rmse:.4f}, "
        f"R2={metrics.r2:.4f}",
    )

    return results
