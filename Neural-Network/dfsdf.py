import numpy as np
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
