# Combined TensorFlow Notebook: Classification and Regression Models

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import SGD, Adam

# Dummy input/output placeholders
# Replace X, Y with your actual data
X = tf.random.normal((100, 10))
Y_binary = tf.random.uniform((100,), maxval=2, dtype=tf.int32)  # Binary: 0 or 1
Y_multiclass = tf.random.uniform((100,), maxval=10, dtype=tf.int32)  # Multi-class: 0-9
Y_regression = tf.random.normal((100,))  # Continuous targets

# 1. Binary Classification with Binary Cross-Entropy loss:
binary_model = Sequential([
    Dense(units=25, activation='sigmoid'),
    Dense(units=15, activation='sigmoid'),
    Dense(units=1, activation='linear')
])
binary_model.compile(
    loss=BinaryCrossentropy(from_logits=True),
    optimizer='sgd'
)
binary_model.fit(X, Y_binary, epochs=100)
logit_binary = binary_model(X)
pred_binary = tf.nn.sigmoid(logit_binary)



# 2. Multi-Class Classification with Softmax
multiclass_model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=10, activation='linear')
])
multiclass_model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer='sgd'
)
multiclass_model.fit(X, Y_multiclass, epochs=100)
logits_multi = multiclass_model(X)
pred_multi = tf.nn.softmax(logits_multi)

# 3. Multi-Class Classification with Adam and Softmax
adam_model = Sequential([
    Dense(units=25, activation='sigmoid'),
    Dense(units=15, activation='sigmoid'),
    Dense(units=10, activation='linear')
])
adam_model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=SparseCategoricalCrossentropy(from_logits=True)
)
adam_model.fit(X, Y_multiclass, epochs=100)

# 4. Lenear Regression
regression_model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=1, activation='linear')
])
regression_model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss=MeanSquaredError()
)
regression_model.fit(X, Y_regression, epochs=100)
pred_reg = regression_model(X)
