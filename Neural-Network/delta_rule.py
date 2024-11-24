import numpy as np
import time
import matplotlib.pyplot as plt

# Define inputs and target for NOR gate
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([1, 0, 0, 0])
lr = 0.1  # Learning rate


def error(output, target): return target - output


def delta_theta(lr, error): return -1 * lr * error


def delta_w(lr, error, x): return lr * error * x


# Online Training
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
        delta_t = 0  # Renamed to avoid shadowing

        for input, target_value in zip(inputs, target):
            # Calculate net input (sigma) and output
            sigma = (input[0] * w1) + (input[1] * w2)
            output = 1 if sigma >= theta else 0

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
w1_online = np.random.random()
w2_online = np.random.random()
theta_online = np.random.random()

# Reinitialize weights and threshold for batch training
w1_batch = np.random.random()
w2_batch = np.random.random()
theta_batch = np.random.random()

print("Online Training")
w1_online, w2_online, theta_online = online_training(inputs, target, lr, w1_online, w2_online, theta_online)
print(f"Final Weights: w1 = {w1_online}, w2 = {w2_online}, Threshold (theta) = {theta_online}")
print("\nBatch Training")
w1_batch, w2_batch, theta_batch = batch_training(inputs, target, lr, w1_batch, w2_batch, theta_batch)
print(f"Final Weights: w1 = {w1_batch}, w2 = {w2_batch}, Threshold (theta) = {theta_batch}")

