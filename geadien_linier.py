import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def plot_regression(x, y, a, b):
    plt.scatter(x, y, color="blue", label="Data points")
    x_line = range(min(x), max(x) + 1)
    y_line = [a * xi + b for xi in x_line]
    plt.plot(x_line, y_line, color="red", label="Best fit line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression: Best Fit Line")
    plt.legend()
    plt.grid()
    plt.show()

# Combine data and fit model
data = np.array([(1, 5), (5, 4), (3, 3), (4, 9), (9, 5), (7, 8), (6, 1), (2, 2)])
X = data[:, 0].reshape(-1, 1)
y = data[:, 1]
model = LinearRegression(0.01, 1000)
model.fit(X, y)

# Predict and plot
print(model.predict(np.array([10]).reshape(-1, 1)))
plot_regression(X.flatten(), y, model.weights[0], model.bias)