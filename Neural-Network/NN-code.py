import numpy as np
import time
import matplotlib.pyplot as plt

'Q N0. 1'
# Define inputs and target for NOR gate
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([1, 0, 0, 0])
lr = 0.1  # Learning rate


def error(output, target):
    return target - output


def delta_theta(lr, error):
    return -1 * lr * error


def delta_w(lr, error, x):
    return lr * error * x


# Initialize weights and threshold randomly
w1 = np.random.random()
w2 = np.random.random()
theta = np.random.random()
'Online Training'


def online_training(inputs, target, lr, w1, w2, theta):
    epoch = 1

    while True:
        total_error = 0
        for input, target_value in zip(inputs, target):
            # Calculate net input (sigma) and output
            sigma = (input[0] * w1) + (input[1] * w2)
            output = 1 if sigma >= theta else 0

            # Calculate error
            e = error(output, target_value)
            total_error += abs(e)

            # Update weights and threshold
            delta_w1 = delta_w(lr, e, input[0])
            delta_w2 = delta_w(lr, e, input[1])
            w1 += delta_w1
            w2 += delta_w2
            theta += delta_theta(lr, e)

        # Check if total error is zero (convergence)
        if total_error == 0:
            print(f"Training completed in {epoch} epochs.")
            break

        epoch += 1

    return w1, w2, theta


# Run the training function and print final weights and threshold
w1, w2, theta = online_training(inputs, target, lr)
print(f"Final Weights: w1 = {w1}, w2 = {w2}, Threshold (theta) = {theta}")

'Batch Training'


def batch_training(inputs, target, lr, w1, w2, theta):
    epoch = 1
    while True:
        total_error = 0
        for input, target_value in zip(inputs, target):
            # Calculate net input (sigma) and output
            sigma = (input[0] * w1) + (input[1] * w2)
            output = 1 if sigma >= theta else 0

            # Calculate error
            e = error(output, target_value)
            total_error += abs(e)

            # Update weights and threshold
            delta_w1 = delta_w(lr, e, input[0])
            delta_w2 = delta_w(lr, e, input[1])
            w1 += delta_w1
            w2 += delta_w2
            theta += delta_theta(lr, e)

        # Check if total error is zero (convergence)
        if total_error == 0:
            print(f"Training completed in {epoch} epochs.")
            break
