import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_file(path):
    df = pd.read_csv(path)
    # Use just the numeric features (e.g., Age and EstimatedSalary)
    X = df[['Age', 'EstimatedSalary']].values
    y = df['Purchased'].values
    return X, y

def compute_gradient(X, y, y_pred):
    m = len(y)
    dW = (1 / m) * np.dot(X.T, (y_pred - y))
    dB = (1 / m) * np.sum(y_pred - y)
    return dW, dB

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, W, B):
    z = np.dot(X, W) + B
    return sigmoid(z)

def compute_cost_function(y, y_pred):
    m = len(y)
    cost = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return cost

def update_W_B(W, B, dW, dB, learning_rate):
    W = W - learning_rate * dW
    B = B - learning_rate * dB
    return W, B

def normalize(X):
    # Normalizes each feature in X by subtracting the mean and dividing by the standard deviation
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def boundry(y_pred):
    return np.where(y_pred >= 0.5, 1, 0)
 
    
def train(path, epochs=1000, learning_rate=0.001):
    # An epoch is one full pass through the entire training dataset during model training.
    # So if you have 100 data points, one epoch = the model sees all 100 once.
    X, y = read_file(path)
    X = normalize(X)
    m, n = X.shape

    W = np.random.randn(n) * 0.01  
    B = np.random.rand() * 0.01 
   
    for epoch in range(epochs):
        y_pred = predict(X, W, B)
        cost = compute_cost_function(y, y_pred)
        dW, dB = compute_gradient(X, y, y_pred)
        W, B = update_W_B(W, B, dW, dB, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Cost: {cost:.4f}")
    
    return X, y, W, B


def plot_2d_classification(X, y, y_pred, W, B):
    plt.figure(figsize=(8, 6))

    # Actual points
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0', alpha=0.5)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='green', label='Class 1', alpha=0.5)

    # Wrong predictions
    wrong = y != y_pred
    plt.scatter(X[wrong][:, 0], X[wrong][:, 1], color='red', marker='x', label='Wrong Prediction')

    # Decision boundary line
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_vals = -(W[0] * x_vals + B) / W[1]  # solve w1*x + w2*y + b = 0 for y
    plt.plot(x_vals, y_vals, color='black', linestyle='--', label='Decision Boundary')

    plt.xlabel("Feature 1 (Age)")
    plt.ylabel("Feature 2 (Salary)")
    plt.title("Logistic Regression Decision Boundary")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = '/Users/ousmanbah/Desktop/Coursera AI /LOGISTIC REGRESSION/Social_Network_Ads.csv'

    X, y, W, B = train(path)
    y_prob = predict(X, W, B)
    y_pred = boundry(y_prob)

    plot_2d_classification(X, y, y_pred, W, B)