import random
import matplotlib.pyplot as plt

x = [random.randint(500, 3000) for _ in range(50)]

# SCALE the sizes
x = [xi / 1000 for xi in x]

# Generate prices (you can keep the same formula for y)
y = [(0.3 * size) + 0.5 + random.uniform(-0.01, 0.01) for size in x]


def predict(x, w, b):

    # y = mx+b choose any m and b and start with that 
    predicted_vals = []
    
    for i in range(len(x)):
        predicted_vals.append((w*x[i])+b)
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
    
    # DW = average of ( prediction - true value) * X
    # DV = average of ( predicion - true value)
    
    m = len(x)
    dw = 0
    db = 0
    y_pred_list = predict(x, w, b)
    for i in range(m):
       
        dw += (y_pred_list[i] - y_true[i]) * x[i]
        db += (y_pred_list[i] - y_true[i])
    
    dw /= m
    db /= m
    return dw, db


def update_parameters(w, b, dw, db, learning_rate):
    
    # Temp_w = w - (learning_rate * dw) same as b too
    
    temp_w = w - (learning_rate * dw)
    temp_b = b - (learning_rate * db)
    
    w = temp_w
    b = temp_b
    
    return w , b 


def train_model(x, y, learning_rate, iterations):
    
    #  start with prediction fixed w and b values
    w = 5
    b = 3 
    
    #  Then next compute the gradient to start with for dw and db like slope of the first position
    for i in range(iterations):
        
        dw, db = compute_gradients(x, y, w, b)
        
        # update with your learning rate to make best gradient 
        w, b = update_parameters(w, b, dw, db, learning_rate)
        
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

train_model(x,y,0.1,1200)

import matplotlib.pyplot as plt

# After training
w, b = train_model(x, y, learning_rate=0.1, iterations=1200)

# Step 1: Plot the original data points
plt.scatter(x, y, color='blue', label='Actual data')

# Step 2: Plot the prediction line
# Generate predictions using the learned w and b
y_predicted = [w * xi + b for xi in x]
plt.plot(x, y_predicted, color='red', label='Prediction line')

# Step 3: Add labels and title
plt.xlabel('House size (scaled)')
plt.ylabel('Price ($1000s)')
plt.title('House Price Prediction')

# Step 4: Show legend
plt.legend()

# Step 5: Show plot
plt.show()
