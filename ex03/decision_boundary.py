"""Exercise 3: the decision boundary of a Logistic Regression.

In one dimension the model fits a sigmoid on the data and the class is given
by the side of the 0.5 probability line the point falls on. In two
dimensions, that line becomes a boundary that separates the plane.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------- 1 dimension
X, y = make_classification(
    n_samples=100,
    n_features=1,
    n_informative=1,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=1,
    weights=[0.5, 0.5],
    flip_y=0.15,
    class_sep=2.0,
    hypercube=True,
    shift=1.0,
    scale=1.0,
    shuffle=True,
    random_state=88
)

# X has the shape (100, 1) and y the shape (100,)
x = X.reshape(1, -1)[0]

# 2. fit the Logistic Regression on the data
classifier = LogisticRegression()
classifier.fit(X, y)


def predict_probability(coefs, X):
    """Probability that X belongs to the class 1.

    coefs is a list that contains a and b: [coef, intercept]
    X is the features set

    Returns probability of X
    """
    coef, intercept = coefs
    probabilities = 1 / (1 + np.exp(-(coef * X.reshape(1, -1)[0] + intercept)))
    return probabilities


def predict_class(coefs, X):
    """The predicted class: 1 when the probability is above 0.5, else 0."""
    return (predict_probability(coefs, X) > 0.5).astype(int)


coefs = [classifier.coef_[0][0], classifier.intercept_[0]]

# 4. and 5. the same results as Scikit-learn, computed by hand
probabilities = predict_probability(coefs, X)
classes = predict_class(coefs, X)

same_probabilities = np.allclose(probabilities, classifier.predict_proba(X)[:, 1])
same_classes = np.array_equal(classes, classifier.predict(X))

# ---------------------------------------------------------------- 2 dimensions
X_2d, y_2d = make_classification(n_features=2,
                                 n_redundant=0,
                                 n_samples=250,
                                 n_classes=2,
                                 n_clusters_per_class=1,
                                 flip_y=0.05,
                                 class_sep=3,
                                 random_state=43)

# 7. the model learns a boundary that separates the two classes
clf = LogisticRegression()
clf.fit(X_2d, y_2d)

if __name__ == "__main__":
    # 1. the data alone
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title("X (1 dimension) and y")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    fig.savefig(HERE / "scatter_1d.png")

    print("2. Intercept: ", classifier.intercept_)
    print("   Coefficient: ", classifier.coef_, end="\n\n")

    # 3. the data, the fitted sigmoid and the 0.5 probability line
    x_sorted = np.sort(x)
    sigmoid = predict_probability(coefs, x_sorted.reshape(-1, 1))

    fig, ax = plt.subplots()
    ax.scatter(x, y, label="y")
    ax.plot(x_sorted, sigmoid, color="orange", label="fitted sigmoid")
    ax.axhline(0.5, color="red", label="probability = 0.5")
    ax.set_title("X (1 dimension) and y, with the fitted sigmoid")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()
    fig.savefig(HERE / "sigmoid_1d.png")

    print("4. same probabilities as predict_proba:", same_probabilities)
    print("5. same classes as predict:", same_classes, end="\n\n")

    # 6. the same plot with the predicted class, shifted to stay readable
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="y")
    ax.plot(x_sorted, sigmoid, color="orange", label="fitted sigmoid")
    ax.axhline(0.5, color="red", label="probability = 0.5")
    ax.scatter(x, np.where(classes == 1, 0.9, 0.1), marker="x", color="green",
               label="predicted class (shifted)")
    ax.set_title("X (1 dimension), y and the predicted class")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()
    fig.savefig(HERE / "predicted_class_1d.png")

    # 7. the decision boundary in 2 dimensions
    xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                          vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(-5, 5), ylim=(-5, 5),
           xlabel="$X_1$", ylabel="$X_2$")
    f.savefig(HERE / "decision_boundary_2d.png")

    print("7. 2 dimensions - Intercept: ", clf.intercept_)
    print("   Coefficients: ", clf.coef_)

    plt.show()
