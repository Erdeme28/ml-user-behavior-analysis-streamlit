import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

from src.evaluation.metrics import (
    compute_classification_metrics,
    ClassificationMetrics,
)
from src.utils.visualization import save_confusion_heatmap


def run_classification(base_df, delta_df) -> list[ClassificationMetrics]:
    """
    Runs classification experiments for the adopter label.
    Focus: comparison of sampling strategies, scaling, and F1-score reporting.
    """
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

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        stratify=labels,
        test_size=0.2,
        random_state=42,
    )

    print(
        "\n[Classification] Class distribution (train):",
        np.bincount(y_train),
    )

    results: list[ClassificationMetrics] = []

    sampler = SMOTE(random_state=42)
    X_sm, y_sm = sampler.fit_resample(X_train, y_train)

    model = LogisticRegression(solver="liblinear")
    model.fit(X_sm, y_sm)

    predictions = model.predict(X_test)
    metrics = compute_classification_metrics(
        "Logistic Regression + SMOTE",
        y_test,
        predictions,
    )
    results.append(metrics)
    save_confusion_heatmap(
        metrics.confusion_matrix,
        metrics.name,
        "cm_logreg_smote.png",
    )
    print(
        f"{metrics.name}: "
        f"acc={metrics.accuracy:.4f}, "
        f"f1={metrics.f1:.4f}",
    )

    sampler_nm = NearMiss()
    X_nm, y_nm = sampler_nm.fit_resample(X_train, y_train)

    model_nm = LogisticRegression(solver="liblinear")
    model_nm.fit(X_nm, y_nm)

    predictions = model_nm.predict(X_test)
    metrics = compute_classification_metrics(
        "Logistic Regression + NearMiss",
        y_test,
        predictions,
    )
    results.append(metrics)
    save_confusion_heatmap(
        metrics.confusion_matrix,
        metrics.name,
        "cm_logreg_nearmiss.png",
    )
    print(
        f"{metrics.name}: "
        f"acc={metrics.accuracy:.4f}, "
        f"f1={metrics.f1:.4f}",
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(X_train_scaled, y_train)

    predictions = model_knn.predict(X_test_scaled)
    metrics = compute_classification_metrics(
        "KNN (scaled) baseline",
        y_test,
        predictions,
    )
    results.append(metrics)
    save_confusion_heatmap(
        metrics.confusion_matrix,
        metrics.name,
        "cm_knn_scaled.png",
    )
    print(
        f"{metrics.name}: "
        f"acc={metrics.accuracy:.4f}, "
        f"f1={metrics.f1:.4f}",
    )

    sampler_knn = SMOTE(random_state=42)
    X_sm_knn, y_sm_knn = sampler_knn.fit_resample(
        X_train_scaled,
        y_train,
    )

    model_knn_sm = KNeighborsClassifier(n_neighbors=5)
    model_knn_sm.fit(X_sm_knn, y_sm_knn)

    predictions = model_knn_sm.predict(X_test_scaled)
    metrics = compute_classification_metrics(
        "KNN (scaled) + SMOTE",
        y_test,
        predictions,
    )
    results.append(metrics)
    save_confusion_heatmap(
        metrics.confusion_matrix,
        metrics.name,
        "cm_knn_smote.png",
    )
    print(
        f"{metrics.name}: "
        f"acc={metrics.accuracy:.4f}, "
        f"f1={metrics.f1:.4f}",
    )

    model_tree = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
    )
    model_tree.fit(X_train, y_train)

    predictions = model_tree.predict(X_test)
    metrics = compute_classification_metrics(
        "Decision Tree baseline",
        y_test,
        predictions,
    )
    results.append(metrics)
    save_confusion_heatmap(
        metrics.confusion_matrix,
        metrics.name,
        "cm_decision_tree.png",
    )
    print(
        f"{metrics.name}: "
        f"acc={metrics.accuracy:.4f}, "
        f"f1={metrics.f1:.4f}",
    )

    model_forest = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model_forest.fit(X_train, y_train)

    predictions = model_forest.predict(X_test)
    metrics = compute_classification_metrics(
        "Random Forest baseline",
        y_test,
        predictions,
    )
    results.append(metrics)
    save_confusion_heatmap(
        metrics.confusion_matrix,
        metrics.name,
        "cm_random_forest.png",
    )
    print(
        f"{metrics.name}: "
        f"acc={metrics.accuracy:.4f}, "
        f"f1={metrics.f1:.4f}",
    )

    return results
