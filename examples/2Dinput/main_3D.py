import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate, Add, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import NonNeg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate training data
def generate_data(num_samples=1000):
    x = np.random.uniform(-1, 1, (num_samples, 2))
    y = np.sum(x**2, axis=1, keepdims=True)
    return x, y

# Generate training and validation data
x_train, y_train = generate_data(2000)
x_val, y_val = generate_data(300)

# Define the model with skip connections
input_layer = Input(shape=(2,), name='INPUT')

# First hidden layer with skip connection
hidden1 = Dense(4, activation='relu', name='FC1')(input_layer)

# Output layer with skip connection
output_layer = Dense(1, name='FC2')(hidden1)
skip_output = Dense(1, use_bias=False, name='SKIP2')(input_layer)
output_layer = Add(name='Z1')([output_layer, skip_output])

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=[tf.keras.metrics.R2Score()])
history = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_val, y_val))

# Visualize the function output and the neural network output as 3D graphs
def plot_3d(x, y, z, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Generate grid data for visualization
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
x_grid, y_grid = np.meshgrid(x, y)
xy_grid = np.stack([x_grid.flatten(), y_grid.flatten()], axis=-1)

# Compute true function values
z_true = np.sum(xy_grid**2, axis=1).reshape(x_grid.shape)

# Compute neural network predictions
z_pred = model.predict(xy_grid).reshape(x_grid.shape)

# Plot the true function
plot_3d(x_grid, y_grid, z_true, 'True function: x^2 + y^2')

# Plot the neural network approximation
plot_3d(x_grid, y_grid, z_pred, 'Neural network approximation')

import json

# Extract the weights as matrices
layer_names = ['FC1', 'FC2','SKIP2']
weights = {layer_name: model.get_layer(layer_name).get_weights() for layer_name in layer_names}

# Save weights to JSON
weights_json = {layer_name: [w.tolist() for w in weight_matrices] for layer_name, weight_matrices in weights.items()}
with open('model_weights_3D.json', 'w') as json_file:
    json.dump(weights_json, json_file)
