"""Exercise 6: One-vs-Rest multi-class classification on the iris data set.

The Logistic Regression is a binary classifier. To handle the 3 classes of
the iris data set, 3 binary classifiers are trained, each one answering "is
it this class or one of the others?". The predicted class is the one whose
classifier returns the highest probability.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

iris = load_iris()

X = pd.DataFrame(data=iris['data'], columns=iris.feature_names)
y = pd.DataFrame(data=iris['target'], columns=['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=43)


def train(X_train, y_train):
    """Train one binary classifier per class (One-vs-Rest)."""
    clf0 = LogisticRegression()
    clf1 = LogisticRegression()
    clf2 = LogisticRegression()

    target = np.ravel(y_train)
    clf0.fit(X_train, target == 0)
    clf1.fit(X_train, target == 1)
    clf2.fit(X_train, target == 2)

    return clf0, clf1, clf2


def predict_one_vs_all(X, clf0, clf1, clf2):
    """The class whose classifier gives the highest probability."""
    probabilities = np.stack([clf0.predict_proba(X)[:, 1],
                              clf1.predict_proba(X)[:, 1],
                              clf2.predict_proba(X)[:, 1]], axis=1)
    classes = np.argmax(probabilities, axis=1)
    return classes


clf0, clf1, clf2 = train(X_train, y_train)
predictions = predict_one_vs_all(X_test, clf0, clf1, clf2)

# the same thing, done by Scikit-learn itself. The parameter
# `multi_class="ovr"` of LogisticRegression no longer exists, One-vs-Rest is
# now done with the OneVsRestClassifier wrapper
sklearn_classifier = OneVsRestClassifier(LogisticRegression())
sklearn_classifier.fit(X_train, np.ravel(y_train))
sklearn_predictions = sklearn_classifier.predict(X_test)

if __name__ == "__main__":
    print("1. the three classifiers are trained on binary targets:")
    print("   clf0: class 1 is 0, class 0 is 1 and 2")
    print("   clf1: class 1 is 1, class 0 is 0 and 2")
    print("   clf2: class 1 is 2, class 0 is 0 and 1", end="\n\n")

    print("2. predicted classes on the test set:", predictions, sep="\n", end="\n\n")
    print("Scikit-learn predictions:", sklearn_predictions, sep="\n", end="\n\n")
    print("same predictions:", np.array_equal(predictions, sklearn_predictions))
    print("accuracy of the One-vs-Rest implemented here:",
          (predictions == np.ravel(y_test)).mean())
