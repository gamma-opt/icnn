import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Add, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

# Generate training data
def generate_data(num_samples=10000):
    x = np.random.uniform(-3, 3, (num_samples))
    y = 2**(-x)
    return x, y

# Generate training and validation data
x_train, y_train = generate_data(2000)
x_val, y_val = generate_data(300)

# Define the model with skip connections
input_layer = Input(shape=(1,), name='INPUT')

# First hidden layer with skip connection
hidden1 = Dense(2, activation='relu', name='FC1')(input_layer)

# Output layer with skip connection
output_layer = Dense(1, name='FC2')(hidden1)
skip_output = Dense(1, use_bias=False, name='SKIP2')(input_layer)
output_layer = Add(name='Z1')([output_layer, skip_output])

model = Model(inputs=input_layer, outputs=output_layer)

# Initialize the optimizer
optimizer = Adam(learning_rate=0.01)

# Training parameters
epochs = 100
batch_size = 32
steps_per_epoch = x_train.shape[0] // batch_size

# Instantiate the loss function
mse_loss = MeanSquaredError()

# Training loop
for epoch in range(epochs):
    for step in range(steps_per_epoch):
        # Get a batch of training data
        x_batch = x_train[step*batch_size:(step+1)*batch_size]
        y_batch = y_train[step*batch_size:(step+1)*batch_size]

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = model(x_batch, training=True)
            # Compute the loss
            loss = mse_loss(y_batch, y_pred)

        # Compute the gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        # Apply the gradients to the optimizer
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Project the hidden layer weights to be non-negative
        for layer in model.layers[2:]:
            if hasattr(layer, 'use_bias') and layer.use_bias:
                weights = layer.get_weights()
                weights[0] = np.maximum(weights[0], 0)  # Project weights to be non-negative
                layer.set_weights(weights)

    # Compute validation loss
    y_val_pred = model(x_val, training=False)
    val_loss = tf.reduce_mean(mse_loss(y_val, y_val_pred))

    print(f'Epoch {epoch+1}, Validation Loss: {val_loss.numpy()}')

# Evaluate the model
y_val_pred = model(x_val, training=False)
val_loss = tf.reduce_mean(mse_loss(y_val, y_val_pred))
print(f'Final Validation Loss: {val_loss.numpy()}')

# Test the model on a new sample
x_test = np.array([[0.5]])
y_test = model(x_test)
print(f'Prediction for {x_test}: {y_test.numpy()}, Expected: {2**(-x_test)}')

# Generate data for visualization
x = np.linspace(-3, 3, 100)
y_true = 2**(-x)

# Compute neural network predictions
y_pred = model.predict(x.reshape(-1, 1))

# Plot the true function
plt.plot(x, y_true, label='True function: 2^(-x)')

# Plot the neural network approximation
plt.plot(x, y_pred, label='Neural network approximation')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('True function vs Neural network approximation')
plt.legend()

# Show the plot
plt.show()


# Iterate through the layers
for i, layer in enumerate(model.layers):
    # Print layer name
    print(f"Layer {i}: {layer.name}")

    # Get layer parameters
    weights = layer.get_weights()

    # Check if layer has parameters
    if weights:
        # Print parameter container sizes
        for w in weights:
            print(f"    Parameter shape: {w.shape}")
            print(f"    All non-negative: {tf.reduce_all(tf.greater_equal(w, 0))}")
    else:
        print("    No parameters")

    # Print use_bias
    if hasattr(layer, 'use_bias'):
        print(f"    use_bias: {layer.use_bias}")
    else:
        print("    No use_bias attribute")

import json
# Extract the weights as matrices
layer_names = ['FC1', 'FC2', 'SKIP2']
weights = {layer_name: model.get_layer(layer_name).get_weights() for layer_name in layer_names}

# Save weights to JSON
weights_json = {layer_name: [w.tolist() for w in weight_matrices] for layer_name, weight_matrices in weights.items()}
with open('model_weights.json', 'w') as json_file:
    json.dump(weights_json, json_file)