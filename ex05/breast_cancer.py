"""Exercise 5: predict breast cancer with a Logistic Regression.

The data is the Wisconsin breast cancer data set: 9 cell measures between 1
and 10, and a class which is 2 (benign) or 4 (malignant). The column names
are not in the file, they come from breast-cancer-wisconsin.names.
"""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parent.parent / "data" / "breast-cancer-wisconsin.data"

COLUMNS = [
    "Sample code number",
    "Clump Thickness",
    "Uniformity of Cell Size",
    "Uniformity of Cell Shape",
    "Marginal Adhesion",
    "Single Epithelial Cell Size",
    "Bare Nuclei",
    "Bland Chromatin",
    "Normal Nucleoli",
    "Mitoses",
    "Class",
]

BENIGN, MALIGNANT = 2, 4

# the missing values are written "?", and the id is used as the index since
# it holds no information on the cancer
df = pd.read_csv(DATA, names=COLUMNS, na_values="?", index_col="Sample code number")
missing_values = df.isna().sum()
df = df.fillna(df.median())

X = df.drop(columns="Class")
y = df["Class"]

# 1. the data set is unbalanced: always predicting Benign would already be
# right 66% of the time, which is the accuracy any model has to beat
proportion_benign = (y == BENIGN).mean()

# 2. stratify keeps that proportion in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=43,
                                                    stratify=y)

# 3. fit on the train set, predict on both sets
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

predictions_train = classifier.predict(X_train)
predictions_test = classifier.predict(X_test)

# the probability of the class Malignant
probabilities_train = classifier.predict_proba(X_train)[:, 1]
probabilities_test = classifier.predict_proba(X_test)[:, 1]

score_train = classifier.score(X_train, y_train)
score_test = classifier.score(X_test, y_test)

# 4. the confusion matrix: [[true benign, false malignant],
#                           [false benign, true malignant]]
confusion_train = confusion_matrix(y_train, predictions_train)
confusion_test = confusion_matrix(y_test, predictions_test)

if __name__ == "__main__":
    print("missing values:", missing_values[missing_values > 0].to_dict())
    print(df.describe().to_string(), end="\n\n")

    print("1. proportion of class Benign:", proportion_benign)
    print("   always predicting Benign gives an accuracy of "
          f"{proportion_benign:.0%}", end="\n\n")

    print("2. proportion of Benign on the train set:", (y_train == BENIGN).mean())
    print("   proportion of Benign on the test set: ", (y_test == BENIGN).mean(),
          end="\n\n")

    print("# Train")
    print("Class prediction on train set:\n", predictions_train[:10], end="\n\n")
    print("Probability prediction on train set:\n", probabilities_train[:10], end="\n\n")
    print("Score on train set:\n", score_train, end="\n\n")

    print("# Test")
    print("Class prediction on test set:\n", predictions_test[:10], end="\n\n")
    print("Probability prediction on test set:\n", probabilities_test[:10], end="\n\n")
    print("Score on test set:\n", score_test, end="\n\n")

    print("4. confusion matrix on the train set:", confusion_train, sep="\n")
    print("   confusion matrix on the test set:", confusion_test, sep="\n", end="\n\n")

    false_malignant = confusion_test[0][1]
    false_negative = confusion_test[1][0]
    print(f"On the test set, {false_negative} malignant tumours are predicted "
          f"benign (false negatives) and {false_malignant} benign tumours are "
          "predicted malignant (false positives).")
    print("The false negatives are the dangerous ones here: a sick patient is "
          "sent home. A false positive only leads to further examinations.")
