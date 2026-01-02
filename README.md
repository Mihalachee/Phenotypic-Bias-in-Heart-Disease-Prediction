# Investigating Gender Bias and Interpretability in Heart Disease Prediction Models

## Description
This repository contains the code and experiments for the Individual Assignment of the **Artificial Intelligence and Society** course (2025/2026).

The project conducts a safety-oriented fairness audit on the UCI Heart Disease dataset using XGBoost. It focuses on:
1.  **Subgroup Reliability:** Analyzing false-negative risks across sex and chest pain phenotypes.
2.  **Explainability:** Using SHAP (Global, Dependence, and Counterfactual analysis) to diagnose model behavior.
3.  **Mitigation:** Comparing decoupled models vs. a "Safety-First" dynamic thresholding strategy.

## Repository Structure
* `Individual_Assignment.ipynb`: The Jupyter Notebook containing the full analysis pipeline.
* `requirements.txt`: List of Python dependencies required to run the project.
* `README.md`: This file.

## Reproducibility

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Installation
It is recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv_FranciscaMihalache

# Activate the environment
# On macOS/Linux:
source venv_FranciscaMihalache/bin/activate
# On Windows:
# venv_FranciscaMihalache\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### 3. Running the Experiments

1. Open the main notebook:
   ```bash
   jupyter notebook Individual_Assignment.ipynb
   ```
2. Run all cells sequentially via **Cell > Run All**.
   * **Note:** The dataset is fetched automatically via the `ucimlrepo` library, so no manual data download is required.
   * **Reproducibility:** A global `random_state=42` is set to ensure results match the report.

## Author
**Francisca Mihalache**
Master in Artificial Intelligence, University of Porto
