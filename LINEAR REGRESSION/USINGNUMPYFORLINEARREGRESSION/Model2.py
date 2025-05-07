
import numpy as np
import matplotlib.pyplot as plt

def read_file(path):
    data = np.loadtxt(path, delimiter=',',skiprows=1)
    X = data[:, [0]]  
    y = data[:, 1]   

    return X, y


def predict(X, W, B):
    return np.dot(X, W) + B


def compute_gradient(X, y, y_pred):
    m = len(y)
    dW = (1 / m) * np.dot(X.T, (y_pred - y))
    dB = (1 / m) * np.sum(y_pred - y)
    return dW, dB


def compute_cost_function(y, y_pred):
    m = len(y)
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
    return cost

def update_W_B(W, B, dW, dB, learning_rate):
    W = W - learning_rate * dW
    B = B - learning_rate * dB
    return W, B

def normalize(X):
    # Normalizes each feature in X by subtracting the mean and dividing by the standard deviation
    
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def train(path, epochs=1100, learning_rate=0.01):
    # An epoch is one full pass through the entire training dataset during model training.
    # So if you have 100 data points, one epoch = the model sees all 100 once.
    X, y = read_file(path)
    
    #  I normalized the data because- to get the answer closer to 0
     
    y = normalize(y)
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



# plt.plot() = connects values with lines
# plt.scatter() = shows only the individual points
def plot_regression(X, y, W, B):
    
    feature_index = 0
    x_feature = X[:, 0]
    y_pred = predict(X, W, B)

    plt.scatter(x_feature, y, color='blue', label='Actual')
    plt.scatter(x_feature, y_pred, color='red', label='Predicted', alpha=0.6)

    plt.xlabel(f"Feature {feature_index}")
    plt.ylabel("y")
    plt.legend()
    plt.title("Linear Regression Fit")
    plt.show()

def plot_price_comparison(y, y_pred):
    # SECOND PLOT PRICE COMPARISM
    
    plt.figure(figsize=(20, 6)) 
    plt.plot(y, label="Actual Price", color="blue", marker='o')
    plt.plot(y_pred, label="Predicted Price", color="red", linestyle='--', marker='x')
    plt.xlabel("House Index")
    plt.ylabel("Price")
    plt.title("Actual vs Predicted Prices")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    # Replace with your file path (CSV format, no headers)
    
    path = '/Users/ousmanbah/Desktop/Coursera AI /LINEAR REGRESSION/USINGNUMPYFORLINEARREGRESSION/housing_data_100.csv'
    # I train the model here and plot it for previous first pred vs trained pred in dots
    X, y, W, B = train(path)

    plot_regression(X, y, W, B)
    
    # Here I use the trained X Y W B values to do the compare thats 
    # Why i just predict based on the output of the train data
    y_pred = predict(X, W, B)
    #  this uses the two y's to compare
    plot_price_comparison(y, y_pred)