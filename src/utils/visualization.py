import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import OUT_CM_DIR


def save_confusion_heatmap(cm, title: str, filename: str) -> str:
    os.makedirs(OUT_CM_DIR, exist_ok=True)

    matrix_df = pd.DataFrame(
        cm.astype(int),
        index=["True 0", "True 1"],
        columns=["Pred 0", "Pred 1"],
    )

    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    output_path = os.path.join(OUT_CM_DIR, filename)
    plt.savefig(output_path)
    plt.close()

    return output_path
