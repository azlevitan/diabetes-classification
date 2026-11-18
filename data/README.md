# Local dataset

Copy the original project CSV to `data/diabetes.csv`. The dataset lives locally and is ignored by Git.

The notebook expects these numeric columns. Column order is flexible and additional columns are ignored.

| Column | Role |
| --- | --- |
| `Pregnancies` | Feature |
| `Glucose` | Feature |
| `BloodPressure` | Feature |
| `SkinThickness` | Feature |
| `Insulin` | Feature |
| `BMI` | Feature |
| `DiabetesPedigreeFunction` | Feature |
| `Age` | Feature |
| `Outcome` | Binary target, with 0 for negative and 1 for positive |

The notebook treats zeros as missing in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`. Zero pregnancies is a valid feature value.

The supplied notebook did not identify the exact source or redistribution terms for its CSV. Use a copy you are authorized to access.
