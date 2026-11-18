# Diabetes Classification

An independent machine learning project that uses three generations of experiments to learn how training-set size, model choice, hyperparameter search, and decision thresholds affect diabetes classification. Each generation starts with a question raised by the previous one. Progress includes better comparisons and clearer tradeoffs, alongside any gains in measured performance.

## Run the notebooks

1. Put your copy of the dataset at `data/diabetes.csv`. See [the data guide](data/README.md) for the required columns.
2. From the repository root, create an environment and launch JupyterLab. The workflow was smoke-tested with Python 3.12.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   python -m jupyterlab
   ```

   In Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`.

3. Open the notebooks in the order below. In each notebook, choose **Run → Restart Kernel and Run All Cells**.

## Generation 1 · Build a baseline and investigate recall

[Open the first notebook](diabetes_classification.ipynb)

**Question.** Can an XGBoost model trained on 80 rows classify the outcome, and does tuning for recall help?

**Experiment.** Treat zeros in five measurement columns as missing, fit median imputation inside the pipeline, and compare baseline XGBoost with a class-weighted model tuned for recall. The current notebook uses a seeded, stratified split and evaluates the same pipeline selected by cross-validation.

**What it taught.** In the original saved run, accuracy stayed at 72.24% and sensitivity fell from 46.59% to 42.17% as specificity rose. The original split had no fixed seed, and the search and final fit used different preprocessing and weights. These historical scores describe that first attempt and can differ from the current rerun. Generation 2 tests the role of training-set size and changes the cutoff explicitly.

| Validation metric | Original baseline | Original tuned model |
| --- | ---: | ---: |
| Accuracy | 72.24% | 72.24% |
| Sensitivity | 46.59% | 42.17% |
| Specificity | 86.79% | 89.29% |
| Precision | 66.67% | 69.08% |
| Negative predictive value | 74.12% | 73.13% |

The original tuned model also reported ROC-AUC of 0.7847 and trapezoidal PR-AUC of 0.6648.

## Generation 2 · Use more training rows and test the cutoff

[Open the second notebook](diabetes_classification_v2.ipynb)

**Question.** How much does more training data help, and how does moving the cutoff change the kinds of errors?

**Experiment.** Compare the same XGBoost settings trained on 80 and 614 rows. Use average precision to select between logistic regression and XGBoost, then select a cutoff by balanced accuracy. Compare all variants on the same 154 held-out rows.

**What it taught.** Training on more rows raised XGBoost's average precision from 0.625 to 0.676. Cross-validation selected logistic regression. Moving its cutoff from 0.50 to 0.3386 caught 10 more positives and added 14 false positives. Full-data XGBoost had slightly higher test average precision than logistic regression on this split. That tradeoff leads to the next question: what changes when every correct classification counts equally?

| Model and cutoff | Average precision | Balanced accuracy | Sensitivity | Specificity | False positives | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost, 80 rows, 0.50 | 0.625 | 0.706 | 61.1% | 80.0% | 20 | 21 |
| XGBoost, full development set, 0.50 | 0.676 | 0.726 | 59.3% | 86.0% | 14 | 22 |
| Logistic regression, 0.50 | 0.673 | 0.727 | 70.4% | 75.0% | 25 | 16 |
| Logistic regression, tuned cutoff | 0.673 | 0.749 | 88.9% | 61.0% | 39 | 6 |

## Generation 3 · Tune systematically for overall accuracy

[Open the third notebook](diabetes_classification_accuracy.ipynb)

**Question.** With a fixed search budget, how much do hyperparameter and cutoff tuning improve overall accuracy?

**Experiment.** Sample 18 XGBoost configurations and compare them with six logistic regression configurations. Score seven cutoffs using the same validation predictions. Each four-fold inner search takes 96 candidate fits. Five outer folds evaluate the selection procedure on the 614 development rows. The earlier 154 test rows are excluded.

**What it taught.** Joint selection produced three more correct predictions than untuned XGBoost. On the same selected models, moving from 0.50 to the tuned cutoff caught nine more positives and added thirteen false positives, yielding four fewer correct predictions. The 0.50 row is a controlled comparison of cutoff effects within this generation.

| Approach | Pooled outer-fold accuracy |
| --- | ---: |
| Majority class | 65.1% |
| Untuned XGBoost | 75.2% |
| Jointly selected model and cutoff | 75.7% |
| Same selected models at 0.50 | 76.4% |

Compare scores within each generation. The experiments use different evaluation rows and objectives, so the percentages across generations cannot establish a direct performance gain. The progression shows how an initial baseline led to controlled comparisons of training size and cutoff effects, then to a reproducible search with an explicit accuracy objective and fit budget.
