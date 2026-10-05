"""Exercise 2: the sigmoid function and two custom sigmoids.

The sigmoid takes any real number and returns a value between 0 and 1, which
is what makes it usable as a probability. The coefficient inside changes how
steep the curve is, the constant shifts it left or right.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).resolve().parent / "sigmoids.png"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid1(x):
    return 1 / (1 + np.exp(-(0.5 * x + 3)))


def sigmoid2(x):
    return 1 / (1 + np.exp(-(5 * x + 11)))


x = np.linspace(-10, 10, 100)

if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.plot(x, sigmoid(x), label="sigmoid")
    ax.plot(x, sigmoid1(x), label="sigmoid1: 0.5x + 3")
    ax.plot(x, sigmoid2(x), label="sigmoid2: 5x + 11")
    ax.axhline(0.5, color="red", label="probability = 0.5")

    ax.set_title("Sigmoid functions")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.savefig(OUTPUT)
    plt.show()
