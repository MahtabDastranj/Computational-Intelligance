import numpy as np
import matplotlib.pyplot as plt

# NOR gate
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([1, 0, 0, 0])
lr = 0.1  # Learning rate


def error(output, target): return target - output


def delta_theta(lr, error): return -1 * lr * error


def delta_w(lr, error, x): return lr * error * x


def net(x1, x2, w1, w2, theta): return (x1 * w1) + (x2 * w2) - theta


# Online Training
def online_training(inputs, target, lr, w1, w2, theta):
    epoch = 1
    while True:
        total_error = 0
        for input, target_value in zip(inputs, target):
            # Calculate net input and output
            output = 1 if net(input[0], input[1], w1, w2, theta) >= 0 else 0

            # Calculate error
            e = error(output, target_value)
            total_error += abs(e)

            # Update weights and threshold
            delta_w1 = delta_w(lr, e, input[0])
            delta_w2 = delta_w(lr, e, input[1])
            w1 += delta_w1
            w2 += delta_w2
            theta += delta_theta(lr, e)
            lr /= 1.001

        # Check if total error is zero (convergence)
        if total_error == 0:
            print(f"Online Training completed in {epoch} epochs.")
            break

        epoch += 1

    return w1, w2, theta


# Batch Training
def batch_training(inputs, target, lr, w1, w2, theta):
    epoch = 1
    while True:
        total_error = 0
        delta_w1 = 0
        delta_w2 = 0
        delta_t = 0

        for input, target_value in zip(inputs, target):
            output = 1 if net(input[0], input[1], w1, w2, theta) >= 0 else 0

            # Calculate error
            e = error(output, target_value)
            total_error += abs(e)

            # Accumulate weight and threshold updates
            delta_w1 += delta_w(lr, e, input[0])
            delta_w2 += delta_w(lr, e, input[1])
            delta_t += delta_theta(lr, e)

        # Apply updates
        w1 += delta_w1
        w2 += delta_w2
        theta += delta_t
        lr /= 1.001

        # Check if total error is zero (convergence)
        if total_error == 0:
            print(f"Batch Training completed in {epoch} epochs.")
            break

        epoch += 1

    return w1, w2, theta


# Initialize weights and threshold randomly
w1_online = np.random.uniform(-1, 1)
w2_online = np.random.uniform(-1, 1)
theta_online = np.random.uniform(-1, 1)

# Initialize weights and threshold for batch training
w1_batch = np.random.uniform(-1, 1)
w2_batch = np.random.uniform(-1, 1)
theta_batch = np.random.uniform(-1, 1)

print("Online Training")
w1_online, w2_online, theta_online = online_training(inputs, target, lr, w1_online, w2_online, theta_online)
print(f"Final Weights: w1 = {w1_online}, w2 = {w2_online}, Threshold (theta) = {theta_online}")
print("\nBatch Training")
w1_batch, w2_batch, theta_batch = batch_training(inputs, target, lr, w1_batch, w2_batch, theta_batch)
print(f"Final Weights: w1 = {w1_batch}, w2 = {w2_batch}, Threshold (theta) = {theta_batch}")

for input, target_value in zip(inputs, target):
    x1, x2 = input
    index = inputs.index(input)
    print("Online Training")
    output_online = 1 if net(x1, x2, w1_online, w2_online, theta_online) > 0 else 0
    print(f'Input: ({x1}, {x2}), Predicted: {output_online}, Target: {target}')
    print("\nBatch Training")
    output_batch = 1 if net(x1, x2, w1_batch, w2_batch, theta_batch) > 0 else 0
    print(f'Input: ({x1}, {x2}), Predicted: {output_batch}, Target: {target}\n')


x1 = np.linspace(-0.5, 1, 150)
x2_online = -(w1_online * x1 + theta_online) / w2_online
x2_batch = -(w1_batch * x1 + theta_batch) / w2_batch

plt.plot(x1, x2_online, label="Online Training")
plt.plot(x1, x2_batch, label="Batch Training")

for input, target_value in zip(inputs, target):
    x1, x2 = input
    color = 'red' if target_value == 0 else 'blue'
    plt.scatter(x1, x2, color=color)

plt.legend
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Discriminating line of the thought perceptron')
plt.grid(True)
plt.show()
