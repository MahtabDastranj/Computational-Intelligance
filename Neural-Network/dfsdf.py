'''import numpy as np
import matplotlib.pyplot as plt

# Define the NOR function
def nor(x1, x2):
    return int(not (x1 or x2))

# Initialize parameters
np.random.seed(42)
eta = 0.1
epochs = 10


# Define the activation function
def step_function(net_input):
    return 1 if net_input >= 0 else 0


# Training data for NOR function
training_data = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 0]
])


def train_perceptron(training_data, epochs, eta, method='online'):
    np.random.seed(42)
    w1, w2 = np.random.rand(2)
    b = np.random.rand()
    theta = 0
    errors = []

    for epoch in range(epochs):
        total_error = 0
        if method == 'online':
            for x1, x2, target in training_data:
                net_input = w1 * x1 + w2 * x2 + b
                output = step_function(net_input)
                error = target - output
                w1 += eta * error * x1
                w2 += eta * error * x2
                b += eta * error
                total_error += abs(error)
        elif method == 'batch':
            w1_update, w2_update, b_update = 0, 0, 0
            for x1, x2, target in training_data:
                net_input = w1 * x1 + w2 * x2 + b
                output = step_function(net_input)
                error = target - output
                w1_update += eta * error * x1
                w2_update += eta * error * x2
                b_update += eta * error
                total_error += abs(error)
            w1 += w1_update
            w2 += w2_update
            b += b_update
        errors.append(total_error)
        eta /= 1.001

    return w1, w2, b, errors


# Train using online method
w1_online, w2_online, b_online, errors_online = train_perceptron(training_data, epochs, eta, method='online')

# Train using batch method
w1_batch, w2_batch, b_batch, errors_batch = train_perceptron(training_data, epochs, eta, method='batch')

# Plotting the decision boundary and data points
x1_vals = np.linspace(-0.1, 1.1, 200)
x2_vals_online = -(w1_online * x1_vals + b_online) / w2_online
x2_vals_batch = -(w1_batch * x1_vals + b_batch) / w2_batch

plt.plot(x1_vals, x2_vals_online, label='Online Training')
plt.plot(x1_vals, x2_vals_batch, label='Batch Training')

for x1, x2, target in training_data:
    color = 'green' if target == 1 else 'red'
    plt.scatter(x1, x2, color=color)

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Decision Boundary of NOR Function (Online vs Batch)')
plt.show()

# Final test
print("Final weights and bias (Online):", w1_online, w2_online, b_online)
for x1, x2, target in training_data:
    net_input = w1_online * x1 + w2_online * x2 + b_online
    output = step_function(net_input)
    print(f'Input: ({x1}, {x2}), Predicted: {output}, Target: {target}')

print("Final weights and bias (Batch):", w1_batch, w2_batch, b_batch)
for x1, x2, target in training_data:
    net_input = w1_batch * x1 + w2_batch * x2 + b_batch
    output = step_function(net_input)
    print(f'Input: ({x1}, {x2}), Predicted: {output}, Target: {target}')
'''
import numpy as np
import matplotlib.pyplot as plt

# NOR gate inputs and targets
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([1, 0, 0, 0])  # Expected outputs for the NOR gate
lr = 0.1  # Learning rate


def net(input, weights, theta):
    """
    Compute the weighted sum (net input) for the perceptron.
    """
    return np.dot(input, weights) - theta


def batch_training(inputs, targets, lr, weights, theta, max_epochs=1000):
    """
    Train a perceptron using batch gradient descent for a NOR gate.

    Parameters:
        inputs: Input samples.
        targets: Target outputs.
        lr: Learning rate.
        weights: Initial weights.
        theta: Initial threshold (bias).
        max_epochs: Maximum number of epochs to prevent infinite loops.

    Returns:
        Trained weights and threshold.
    """
    epoch = 1
    while epoch <= max_epochs:
        total_error = 0
        weight_updates = np.zeros_like(weights)  # Accumulate weight updates
        theta_update = 0  # Accumulate threshold updates

        # Iterate over all samples
        for input, target in zip(inputs, targets):
            # Calculate net input and continuous output
            sigma = net(input, weights, theta)
            output = 1 if sigma >= 0 else 0

            # Compute error
            error = target - output
            total_error += abs(error)

            # Accumulate updates for weights and threshold
            weight_updates += lr * error * input
            theta_update -= lr * error  # Subtract since theta is in the opposite direction

        # Apply accumulated updates
        weights += weight_updates
        theta += theta_update

        print(f"Epoch {epoch}: Total Error = {total_error}")

        # Check for convergence
        if total_error == 0:
            print(f"Batch Training completed in {epoch} epochs.")
            break

        epoch += 1

    return weights, theta


# Initialize weights and threshold randomly
np.random.seed(42)  # For reproducibility
weights = np.random.uniform(-0.1, 0.1, size=2)  # Two weights (one for each input)
theta = np.random.uniform(-0.1, 0.1)  # Threshold (bias)

# Train the perceptron using batch gradient descent
print("Batch Training for NOR Gate")
trained_weights, trained_theta = batch_training(inputs, targets, lr, weights, theta)
print(f"Trained Weights: {trained_weights}")
print(f"Trained Threshold: {trained_theta}")

# Test the trained perceptron
print("\nTesting the NOR Gate Perceptron:")
for input, target in zip(inputs, targets):
    sigma = net(input, trained_weights, trained_theta)
    output = 1 if sigma >= 0 else 0
    print(f"Input: {input}, Predicted: {output}, Target: {target}")

# Plot the decision boundary
x1 = np.linspace(-0.5, 1.5, 150)
x2 = -(trained_weights[0] * x1 - trained_theta) / trained_weights[1]

plt.plot(x1, x2, label="Decision Boundary")
for input, target in zip(inputs, targets):
    color = 'blue' if target == 1 else 'red'
    plt.scatter(input[0], input[1], color=color)

plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision Boundary for NOR Gate Perceptron')
plt.grid(True)
plt.show()

