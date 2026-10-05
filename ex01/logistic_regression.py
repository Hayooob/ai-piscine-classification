"""Exercise 1: classify data with a Logistic Regression from Scikit-learn."""
from sklearn.linear_model import LogisticRegression

X = [[0], [0.1], [0.2], [1], [1.1], [1.2], [1.3]]
y = [0, 0, 0, 1, 1, 1, 0]

classifier = LogisticRegression()
classifier.fit(X, y)

x_pred = [[0.5]]

# 1. the predicted class
predicted_class = classifier.predict(x_pred)

# 2. the probability to belong to each class
predicted_probabilities = classifier.predict_proba(x_pred)

# 3. what the model learnt, and its accuracy on the data it learnt on
coefficient = classifier.coef_
intercept = classifier.intercept_
score = classifier.score(X, y)

if __name__ == "__main__":
    print("1. predicted class:", predicted_class)
    print("2. predicted probabilities:", predicted_probabilities[0])
    print("3. Coefficient:", coefficient, sep="\n")
    print("Intercept:", intercept, sep="\n")
    print("Score:", score, sep="\n")
