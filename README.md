# Investigating Gender Bias and Interpretability in Heart Disease Prediction Models

Code and experiments for the Individual Assignment of the **Artificial Intelligence and Society** course (2025/2026).
This project performs a **safety-oriented fairness audit** on the UCI Heart Disease dataset using **XGBoost**, focusing on subgroup reliability and interpretability.

## Scope (Read this first)
- This is an **academic fairness/safety audit** on historical data.
- **Not a medical device** and **not for clinical use**.

## Project Goals

1. **Subgroup Reliability**
   - Analyze **false-negative risk** across **sex** and **chest pain (cp) phenotypes**.
2. **Explainability**
   - Use **SHAP** for global importance and dependence analysis.
   - Use **counterfactual-style** probing (as implemented in the notebook) to diagnose model behavior.
3. **Mitigation**
   - Compare:
     - **Decoupled models** (separate classifiers per sex)
     - **Safety-first thresholds** (phenotype-aware thresholding prioritizing recall for high-risk groups)

## Repository Structure
- `Individual_Assignment.ipynb` — end-to-end analysis (data → training → audit → SHAP → mitigation)
- `requirements.txt` — Python dependencies
- `README.md` — project overview and run instructions

## Dataset
- Source: UCI Heart Disease dataset retrieved via `ucimlrepo` (no manual download).
- Note: internet access is required at first run to fetch the dataset.

## Reproducibility Notes
- The notebook uses `random_state=42` where applicable.
- Exact metric values may vary slightly across OS/CPU and library versions (e.g., XGBoost + SHAP).
- For closest reproducibility, use pinned dependency versions and keep CPU settings consistent.

## Setup
### 1) Prerequisites
- Python **3.10+** (tested with Python **3.11**)

### 2) Install
Recommended: virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

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
