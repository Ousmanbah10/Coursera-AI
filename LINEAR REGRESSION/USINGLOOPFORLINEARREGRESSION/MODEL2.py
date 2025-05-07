import random
import matplotlib.pyplot as plt

x = [
    [1.2, 3, 10],  # House 1: 1200 sqft, 3 bedrooms, 10 years old
    [2.5, 4, 5],   # House 2: 2500 sqft, 4 bedrooms, 5 years old
    [0.9, 2, 20],  # House 3: 900 sqft, 2 bedrooms, 20 years old
    [1.8, 3, 15],  # House 4: 1800 sqft, 3 bedrooms, 15 years old
    [2.2, 5, 8],   # House 5: 2200 sqft, 5 bedrooms, 8 years old
    [1.5, 2, 12],  # House 6: 1500 sqft, 2 bedrooms, 12 years old
    [2.0, 4, 7],   # House 7: 2000 sqft, 4 bedrooms, 7 years old
    [1.1, 2, 18]   # House 8: 1100 sqft, 2 bedrooms, 18 years old
]
y = [
    3.5,  # Price for House 1
    5.5,  # Price for House 2
    2.8,  # Price for House 3
    4.0,  # Price for House 4
    6.0,  # Price for House 5
    3.2,  # Price for House 6
    5.0,  # Price for House 7
    2.9   # Price for House 8
]

def predict(x, w, b):
    predicted_vals = []
    for features in x:
        value = 0
        for j in range(len(features)):
            value += w[j] * features[j]
        predicted_vals.append(value + b)  # This should be inside the loop
    return predicted_vals

def compute_cost(y_true, y_pred):
    
    # Cost function : total cost = (y predcit - y_correct) ** 2 
    #  Cost function = 1* totalcost / 2M 
    total_cost = 0
    m = len(y_true)
    for i in range(m):
        total_cost += (y_pred[i] - y_true[i]) ** 2
        
    return total_cost / (2 * m)



def compute_gradients(x, y_true, w, b):
    m = len(x)
    n_features = len(x[0])
    dw = [0] * n_features
    db = 0

    y_pred_list = predict(x, w, b)

    for i in range(m):
        error = y_pred_list[i] - y_true[i]
        
        for j in range(n_features):
            dw[j] += error * x[i][j]
        
        db += error

    dw = [d / m for d in dw]
    db /= m
    
    return dw, db

def update_gradient(w, b, dw, db, learning_rate):
    # w and dw are lists
    
    temp_w = []
    for i in range(len(w)):
        temp_w.append(w[i] - learning_rate * dw[i])
    
    temp_b = b - (learning_rate * db)
    
    w = temp_w
    b = temp_b
    
    return w, b


def train_model(x, y, learning_rate, iterations):
    
    #  start with prediction fixed w and b values
    n_features = len(x[0])
    w = [0] * n_features
    b = 0 
    
    
    #  Then next compute the gradient to start with for dw and db like slope of the first position
    for i in range(iterations):
        
        dw, db = compute_gradients(x, y, w, b)
        
        # update with your learning rate to make best gradient 
        w, b = update_gradient(w, b, dw, db, learning_rate)
        
         # ALWAYS compute cost
        y_pred_list = predict(x, w, b)
        cost = compute_cost(y, y_pred_list)
        
        # Only print every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost}")
        
        # Check if cost is low enough
        if cost < 0.001:
            print(f"Cost {cost} is low enough, stopping early at iteration {i}!")
            break    
     
    print(w,b)
    return w, b



w, b = train_model(x, y, learning_rate=0.01, iterations=1200)

# Get sizes (first feature)
sizes = [xi[0] for xi in x]

# Get predictions
y_predicted = predict(x, w, b)

# Sort by size
sorted_pairs = sorted(zip(sizes, y_predicted))  # pairs of (size, predicted_price)

# Unzip sorted pairs
sorted_sizes, sorted_preds = zip(*sorted_pairs)

print(sorted_sizes,"hey," ,sorted_preds)
# Plot correctly
plt.scatter(sizes, y, color='blue', label='Actual data')
plt.plot(sorted_sizes, sorted_preds, color='red', label='Prediction line')

plt.xlabel('House Size (1000 sqft)')
plt.ylabel('Price ($100,000s)')
plt.title('House Price Prediction')
plt.legend()
plt.show()
