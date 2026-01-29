from dataclasses import dataclass
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


@dataclass
class ClassificationMetrics:
    name: str
    accuracy: float
    f1: float
    confusion_matrix: np.ndarray


@dataclass
class RegressionMetrics:
    name: str
    mae: float
    rmse: float
    r2: float


def compute_classification_metrics(
    name: str,
    y_true,
    y_pred,
) -> ClassificationMetrics:
    matrix = confusion_matrix(y_true, y_pred)
    return ClassificationMetrics(
        name=name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred)),
        confusion_matrix=matrix,
    )


def compute_regression_metrics(
    name: str,
    y_true,
    y_pred,
) -> RegressionMetrics:
    mse = mean_squared_error(y_true, y_pred)
    return RegressionMetrics(
        name=name,
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mse)),
        r2=float(r2_score(y_true, y_pred)),
    )
