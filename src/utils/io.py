import os

from src.config import OUT_DIR, OUT_CM_DIR, DATA_DIR
from src.evaluation.metrics import ClassificationMetrics, RegressionMetrics


def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_CM_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def save_metrics_summary(
    cls_summary: list[ClassificationMetrics],
    reg_summary: list[RegressionMetrics],
    out_path: str,
) -> None:
    lines = []

    lines.append("=== CLASSIFICATION SUMMARY ===\n")
    for metrics in cls_summary:
        lines.append(
            f"- {metrics.name}: "
            f"accuracy={metrics.accuracy:.4f}, "
            f"f1={metrics.f1:.4f}, "
            f"confusion_matrix={metrics.confusion_matrix.tolist()}\n"
        )

    lines.append("\n=== REGRESSION SUMMARY ===\n")
    for metrics in reg_summary:
        lines.append(
            f"- {metrics.name}: "
            f"MAE={metrics.mae:.4f}, "
            f"RMSE={metrics.rmse:.4f}, "
            f"R2={metrics.r2:.4f}\n"
        )

    with open(out_path, "w", encoding="utf-8") as file:
        file.writelines(lines)
