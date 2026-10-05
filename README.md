# Classification

Solutions for the Classification day of the AI piscine: logistic regression,
the sigmoid, the decision boundary, a stratified train test split, the breast
cancer prediction and the One-vs-Rest multi-class approach.

## Structure

```
.
├── data/
│   ├── breast-cancer-wisconsin.data
│   └── breast-cancer-wisconsin.names
├── ex00/
│   ├── requirements.txt
│   └── Notebook_ex00.ipynb
├── ex01/logistic_regression.py
├── ex02/sigmoid.py                 -> sigmoids.png
├── ex03/decision_boundary.py       -> scatter_1d.png, sigmoid_1d.png,
│                                      predicted_class_1d.png,
│                                      decision_boundary_2d.png
├── ex04/train_test_split_classification.py
├── ex05/breast_cancer.py
└── ex06/multi_class.py
```

## Virtual environment (ex00)

```bash
python -m venv env/ex00
source env/ex00/Scripts/activate      # Windows (Git Bash); Linux/macOS: source env/ex00/bin/activate
pip install -r ex00/requirements.txt
python --version
python -c "import jupyter, numpy, pandas, matplotlib, sklearn"
```

Tested with Python 3.13, scikit-learn 1.9.1, pandas 3.0.6, NumPy 2.5.3 and
matplotlib 3.11.2.

## Data

`breast-cancer-wisconsin.data` and its description come from the
[01-edu subject](https://github.com/01-edu/public/tree/master/subjects/ai/classification/data).
The file has no header, the column names are taken from the `.names` file.

## Running an exercise

```bash
python ex05/breast_cancer.py
```

## Results

- **ex01** — predicted class `0`, probabilities `[0.61450206 0.38549794]`,
  coefficient `[[0.81793264]]`, intercept `[-0.87524276]`, score
  `0.7142857142857143`.
- **ex03** — in one dimension: intercept `[-0.98497516]`, coefficient
  `[[1.18885765]]`. The hand written `predict_probability` and
  `predict_class` return exactly what `predict_proba` and `predict` return.
- **ex04** — without `stratify`, the train set holds 12.5% of class 1 and the
  test set 100%: the model is evaluated on a class it barely saw. With
  `stratify`, both sets hold 30%.
- **ex05** — 65.5% of the tumours are benign, so always answering "benign"
  would already be right 66% of the time. The model reaches **96.96%** on the
  train set and **96.43%** on the test set, well above that baseline.
- **ex06** — the three One-vs-Rest classifiers predict the same classes as
  Scikit-learn, and the accuracy on the test set is 100%.

## Confusion matrices of ex05

Train set, then test set (rows: true benign / true malignant, columns:
predicted benign / predicted malignant):

```
[[357   9]        [[90  2]
 [  8 185]]        [ 3 45]]
```

On the test set, 3 malignant tumours are predicted benign (false negatives)
and 2 benign tumours are predicted malignant (false positives). The false
negatives are the dangerous ones: a sick patient is sent home, while a false
positive only leads to further examinations. For this reason a medical model
is usually tuned to lower the false negatives, even if the false positives
rise.

## Notes

- **ex05** — the missing values are written `?` in the file; they are read as
  NaN and replaced with the median of their column. `Sample code number` is
  set as the index since an id carries no information on the cancer. The
  class stays as in the file: 2 is benign, 4 is malignant.
- **ex06** — the parameter `multi_class="ovr"` of `LogisticRegression` no
  longer exists in recent versions of Scikit-learn; the comparison uses the
  `OneVsRestClassifier` wrapper, which does the same thing.
- the values of ex01 and ex03 differ from the audit after the 4th decimal
  (for example `0.61450206` against `0.61450526`): the solver of
  `LogisticRegression` stops on a slightly different iteration depending on
  its version. Everything in ex05, where the results matter, matches the
  audit exactly.
