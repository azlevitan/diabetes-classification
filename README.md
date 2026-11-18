# Diabetes Classification

An independent machine learning project that uses XGBoost to classify the binary diabetes outcome in a tabular dataset. It compares a baseline classifier with a model tuned for sensitivity, the share of positive cases identified by the model.

## Run the notebook

1. Put your copy of the dataset at `data/diabetes.csv`. See [the data guide](data/README.md) for the required columns.
2. From the repository root, create an environment and launch JupyterLab. The workflow was smoke-tested with Python 3.12.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   python -m jupyterlab
   ```

   In Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`.

3. Open `diabetes_classification.ipynb` and choose **Run → Restart Kernel and Run All Cells**. The grid search evaluates 144 configurations across five folds.

## Experiment

1. Treat zeros in five measurement columns as missing values.
2. Use a seeded, stratified split with 80 training rows, preserving the original notebook's training size. The remaining rows form the validation set.
3. Fit a baseline XGBoost classifier with median imputation. Tune a class-weighted XGBoost pipeline using five-fold cross-validation on the training set, with median imputation fit separately inside each fold.
4. Compare accuracy, sensitivity, specificity, precision, and negative predictive value on the validation set. Inspect the tuned model's ROC, precision-recall, and calibration curves.

## Original notebook results

These numbers were saved in the submitted notebook and are historical. The split had no fixed seed, and the original search and final fit used different preprocessing and class weights. The current notebook uses a seeded split and evaluates the same pipeline selected by the search. Run it with the original CSV to obtain results for the current workflow.

| Validation metric | Original baseline | Original tuned model |
| --- | ---: | ---: |
| Accuracy | 72.24% | 72.24% |
| Sensitivity | 46.59% | 42.17% |
| Specificity | 86.79% | 89.29% |
| Precision | 66.67% | 69.08% |
| Negative predictive value | 74.12% | 73.13% |

The original tuned model reported ROC-AUC of 0.7847 and trapezoidal PR-AUC of 0.6648. Its saved accuracy matched the baseline's, while sensitivity fell and specificity rose. The small training set and single validation split limit what can be concluded about performance on other datasets. This is an educational experiment for classifying the dataset's outcome label.
