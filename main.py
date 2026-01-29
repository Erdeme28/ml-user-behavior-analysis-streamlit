from src.data.data_generator import generate_dataset
from src.models.classification import run_classification
from src.models.regression import run_regression
from src.utils.io import ensure_dirs, save_metrics_summary
from src.config import OUT_DIR, RANDOM_STATE


def main() -> None:
    ensure_dirs()

    base_df, delta_df = generate_dataset(
        n_samples=8000,
        random_state=RANDOM_STATE,
        save_csv=False,
    )

    classification_results = run_classification(base_df, delta_df)
    regression_results = run_regression(base_df, delta_df)

    save_metrics_summary(
        classification_results,
        regression_results,
        out_path=f"{OUT_DIR}/metrics_summary.txt",
    )

    print(
        "\n✅ Done. Check the outputs/ directory for plots "
        "and metrics_summary.txt for the summary."
    )


if __name__ == "__main__":
    main()
