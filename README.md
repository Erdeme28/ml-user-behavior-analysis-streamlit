# ML User Behavior Analysis

## Project Overview

ML User Behavior Analysis is an educational machine learning project that demonstrates a complete ML pipeline for a simulated music platform. The project generates synthetic user data and provides interactive pages to train, evaluate, compare, export, and explain models using a Streamlit front end.

Primary tasks implemented:
- Classification: predict whether a user will become a premium subscriber (binary classification)
- Regression: estimate a continuous user engagement score

All components run locally through a Streamlit application without requiring a separate web backend.

## Goals

- Demonstrate end-to-end ML workflows: data generation, preprocessing, model training, evaluation, comparison, and explainability.
- Provide an educational, reproducible environment using synthetic data.
- Offer an interactive interface that allows users to experiment with algorithms and settings.

## Features

- Synthetic data generator for user behavior simulation
- Classification algorithms: Logistic Regression, KNN, Decision Tree, Random Forest
- Regression algorithms: Linear Regression, KNN Regressor, Random Forest Regressor
- Imbalance handling options for classification: none, SMOTE, NearMiss
- Evaluation metrics and visualizations (confusion matrices, prediction vs ground truth plots, residuals)
- Model comparison and experiment export
- Explainability via feature importance for tree-based models

## Technologies

- Python 3.8+ (recommended)
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Seaborn

See `requirements.txt` for exact pinned dependencies used by the project.

## Quick Start

1. Create a Python virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1  ;# PowerShell usage on Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application from the repository root:

   ```bash
   streamlit run app\1_ML_User_Behavior_Analysis.py
   ```

On Windows PowerShell you may run these commands from the project root directory.

There is also a `run_app.bat` file included for convenience that attempts to start the app with the environment's Python.

## Project Structure

Top-level layout (key files and folders):

```
- app/
  - 1_ML_User_Behavior_Analysis.py  (Streamlit entry page)
  - pages/
    - 2_Classification.py
    - 3_Regression.py
    - 4_Model_Comparison.py
    - 5_Export_Results.py
    - 6_Explainability.py

- src/
  - config.py
  - data/
    - data_generator.py
  - models/
    - classification.py
    - regression.py
  - evaluation/
    - metrics.py
  - utils/
    - io.py
    - visualization.py

- data/                 (input datasets or sample CSVs if provided)
- outputs/              (generated outputs: metrics, figures, exported results)
- main.py               (optional script; may be used to run or test modules)
- requirements.txt      (Python dependencies)
- run_app.bat           (helper script to start the Streamlit app)
```

Note: The exact layout may contain additional helper files or caches (for example, `__pycache__`).

## Configuration

- `src/config.py` centralizes settings used across the project (random seeds, default model hyperparameters, file paths). Update values there if you want to change default behavior.
- Output files, experiment logs, and exported CSVs are written to the `outputs/` directory by default.

## How to Use the App

1. Start the Streamlit app (see Quick Start).
2. Use the left-hand navigation in Streamlit to move between pages:
   - Home: overview and data generation controls
   - Classification: choose a classifier, preprocessing (scaling, balancing), train and evaluate
   - Regression: choose a regressor, train and evaluate, inspect residuals and sample predictions
   - Model Comparison: run multiple models and view leaderboards
   - Export Results: save trained model metrics and confusion matrices to `outputs/`
   - Explainability: view feature importance and simple explanations for tree-based models

3. Adjust sliders, dropdowns, and other interactive controls to run experiments with different settings.

## Tests and Validation

This repository does not include a formal test suite by default. To validate basic runtime behavior manually:

- Ensure dependencies are installed.
- Run the main Streamlit page and exercise each page's controls.
- Inspect `outputs/` for generated CSVs and images.

If you want, you can add lightweight unit tests for functions in `src/` (for example, data generation and evaluation functions) using `pytest`.

## Troubleshooting

- Streamlit fails to start or imports fail: ensure you installed dependencies into the same Python interpreter you use to start Streamlit (check `python --version` and `pip show streamlit`).
- Missing packages: run `pip install -r requirements.txt` again and watch for errors during installation.
- Permission or path errors when writing to `outputs/`: verify your user has write permissions to the project directory.
- Unexpected model results: confirm that data generation parameters and preprocessing steps are set as intended on the app pages.

## Development Notes and Suggestions

- Add unit tests for critical functions in `src/` (data generation, preprocessing, and evaluation metrics).
- Add a small CI job (GitHub Actions) to run linting and tests on push.
- Consider bundling a `pyproject.toml` and using `tox` or `nox` for consistent test environments.
- If you plan to share this project, add a link to a demo deployment or a short video walkthrough.

## Contributing

Contributions are welcome. For small changes or bugfixes:

1. Fork the repository.
2. Create a feature branch.
3. Make changes and add tests where appropriate.
4. Open a pull request with a clear description of changes.

## License

This project includes a `LICENSE` file at the repository root. Review it for the project's license terms.

## Contact

If you have questions about the project or need help, open an issue in the repository or contact the repository owner as listed on the hosting platform.


---

Last updated: January 2026
