"""Exercise 4: split a classification data set, keeping the classes balanced.

Splitting without shuffling can give a train set and a test set with very
different proportions of each class: the model is then evaluated on something
it has not learnt. `stratify` keeps the proportion of each class.
"""
import numpy as np
from sklearn.model_selection import train_test_split

# 1. a split that goes wrong: the 2 rows of the test set are both of class 1
X = np.arange(1, 21).reshape(10, -1)
y = np.zeros(10)
y[7:] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    shuffle=False)

# 2. the same split, keeping the proportion of class 1 constant
X2 = np.arange(1, 201).reshape(100, -1)
y2 = np.zeros(100)
y2[70:] = 1

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2,
                                                        test_size=0.2,
                                                        random_state=43,
                                                        stratify=y2)

if __name__ == "__main__":
    print("X_train:\n", X_train, end="\n\n\n")
    print("y_train:\n", y_train, end="\n\n\n")
    print("X_test:\n", X_test, end="\n\n\n")
    print("y_test:\n", y_test, end="\n\n")
    print("1. proportion of class 1 on the train set:", y_train.mean())
    print("   proportion of class 1 on the test set: ", y_test.mean(), end="\n\n")

    print("2. with stratify:")
    print("   proportion of class 1 on the train set:", y2_train.mean())
    print("   proportion of class 1 on the test set: ", y2_test.mean())
