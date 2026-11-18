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

## Accuracy-first hyperparameter search

Open [the accuracy-first notebook](diabetes_classification_accuracy.ipynb) to compare 18 randomly sampled XGBoost configurations and six logistic regression configurations. The inner search tests seven cutoffs on the same validation scores and selects the model and cutoff by overall accuracy. Four inner folds require 96 candidate fits per search. Five outer folds evaluate the selection procedure on the development rows and compare it with default XGBoost and a majority-class baseline. Sensitivity and specificity show the effect of optimizing for accuracy.

On the 614 development rows, pooled outer-fold accuracy was 65.1% for the majority baseline, 75.2% for default XGBoost, 75.7% for joint model-and-cutoff selection, and 76.4% when the jointly selected models used a fixed 0.50 cutoff. That last comparison isolates the cutoff on the same selected models. Joint cutoff tuning recovered nine more positive cases and added thirteen false positives relative to 0.50. The earlier 154 test rows were excluded from this comparison.

## Second-generation experiment

Open [the v2 notebook](diabetes_classification_v2.ipynb) in the same environment and run all cells. It reserves 20% of the data for a shared stratified test set and uses the remaining rows for development. It compares an 80-row XGBoost baseline, the same baseline trained on all development rows, and a model selected from regularized logistic regression and shallow XGBoost candidates.

The inner cross-validation search selects the model by average precision. The outer cross-validation loop selects a cutoff by balanced accuracy. The test comparison reports both ranking quality and the effect of the tuned cutoff on false positives and missed cases.

### V2 results on the project CSV

The fixed split used 614 development rows and 154 test rows. Cross-validation selected logistic regression and a cutoff of 0.3386. All values below come from the same test set.

| Model and cutoff | Average precision | Balanced accuracy | Sensitivity | Specificity | False positives | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost, 80 rows, 0.50 | 0.625 | 0.706 | 61.1% | 80.0% | 20 | 21 |
| XGBoost, full development set, 0.50 | 0.676 | 0.726 | 59.3% | 86.0% | 14 | 22 |
| Logistic regression, 0.50 | 0.673 | 0.727 | 70.4% | 75.0% | 25 | 16 |
| Logistic regression, tuned cutoff | 0.673 | 0.749 | 88.9% | 61.0% | 39 | 6 |

Using more rows improved XGBoost's average precision. Lowering the logistic cutoff caught 10 more positives than its default cutoff and added 14 false positives. Full-data XGBoost had slightly higher test average precision than logistic regression. These estimates come from one split of a small dataset.

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
