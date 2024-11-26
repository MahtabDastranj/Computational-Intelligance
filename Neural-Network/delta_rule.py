import numpy as np
import matplotlib.pyplot as plt

# NOR gate inputs and targets
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([1, 0, 0, 0])
lr = 0.1  # Learning rate


def error(output, target):
    return target - output


def delta_theta(lr, error):
    return lr * error


def delta_w(lr, error, x):
    return lr * error * x


def net(x1, x2, w1, w2, theta):
    return x1 * w1 + x2 * w2 - theta


# Online Training
def online_training(inputs, target, lr, w1, w2, theta):
    epoch = 1
    max_epochs = 20
    decision_boundaries = []

    while epoch < max_epochs:
        total_error = 0

        for input, target_value in zip(inputs, target):
            # Calculate output
            output = 1 if net(input[0], input[1], w1, w2, theta) >= 0 else 0

            # Calculate error
            e = error(output, target_value)
            total_error += abs(e)

            # Update weights and threshold
            w1 += delta_w(lr, e, input[0])
            w2 += delta_w(lr, e, input[1])
            theta -= delta_theta(lr, e)

        if w2 != 0:
            x_vals = np.linspace(-0.1, 1.1, 100)
            y_vals = -(w1 * x_vals - theta) / w2
            decision_boundaries.append((x_vals, y_vals))

        if total_error == 0:
            print(f"Online Training completed in {epoch} epochs.")
            print(f'w1 = {w1}, w2 = {w2}, theta = {theta}')
            break

        epoch += 1

    return w1, w2, theta, decision_boundaries


# Batch Training
def batch_training(inputs, target, lr, w1, w2, theta):
    epoch = 1
    max_epochs = 20
    decision_boundaries = []

    while epoch < max_epochs:
        total_error = 0
        delta_w1 = 0
        delta_w2 = 0
        delta_t = 0

        for input, target_value in zip(inputs, target):
            output = 1 if net(input[0], input[1], w1, w2, theta) >= 0 else 0

            e = error(output, target_value)
            total_error += abs(e)

            # Accumulate updates
            delta_w1 += delta_w(lr, e, input[0])
            delta_w2 += delta_w(lr, e, input[1])
            delta_t -= delta_theta(lr, e)

        # Apply updates
        w1 += delta_w1
        w2 += delta_w2
        theta += delta_t

        if w2 != 0:
            x_vals = np.linspace(-0.1, 1.1, 100)
            y_vals = -(w1 * x_vals - theta) / w2
            decision_boundaries.append((x_vals, y_vals))

        if total_error == 0:
            print(f"Batch Training completed in {epoch} epochs.")
            print(f'w1 = {w1}, w2 = {w2}, theta = {theta}')
            break

        epoch += 1

    return w1, w2, theta, decision_boundaries


# Initialize weights and thresholds randomly
np.random.seed(42)
w1_online, w2_online, theta_online = np.random.uniform(-0.1, 0.1, 3)
w1_batch, w2_batch, theta_batch = np.random.uniform(-0.1, 0.1, 3)

# Train perceptrons using online and batch training
print("Training Online...")
w1_online, w2_online, theta_online, online_decision_boundaries = online_training(
    inputs, target, lr, w1_online, w2_online, theta_online
)

print("Training Batch...")
w1_batch, w2_batch, theta_batch, batch_decision_boundaries = batch_training(
    inputs, target, lr, w1_batch, w2_batch, theta_batch
)

# Visualization
fig, axs = plt.subplots(2, 1, figsize=(16, 6))

# Online Training Plot
axs[0].set_title("Online Training")
for i, (x_vals, y_vals) in enumerate(online_decision_boundaries):
    axs[0].plot(x_vals, y_vals, linestyle="--", alpha=0.3, label=f"Epoch {i + 1}")
x_vals = np.linspace(-0.1, 1.1, 100)
y_vals = -(w1_online * x_vals - theta_online) / w2_online
axs[0].plot(x_vals, y_vals, color="red", label="Final Decision Boundary")
for input, target_value in zip(inputs, target):
    color = "blue" if target_value == 1 else "green"
    axs[0].scatter(input[0], input[1], color=color, s=100, label=f"Class {target_value}")

# Batch Training Plot
axs[1].set_title("Batch Training")
for i, (x_vals, y_vals) in enumerate(batch_decision_boundaries):
    axs[1].plot(x_vals, y_vals, linestyle="--", alpha=0.3, label=f"Epoch {i + 1}")
x_vals = np.linspace(-0.1, 1.1, 100)
y_vals = -(w1_batch * x_vals - theta_batch) / w2_batch
axs[1].plot(x_vals, y_vals, color="red", label="Final Decision Boundary")
for input, target_value in zip(inputs, target):
    color = "blue" if target_value == 1 else "green"
    axs[1].scatter(input[0], input[1], color=color, s=100, label=f"Class {target_value}")

for ax in axs:
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True)

plt.tight_layout()
plt.show()

'''for input, target_value in zip(inputs, target):
    x1, x2 = input
    output_online = 1 if net(x1, x2, w1_online, w2_online, theta_online) > 0 else 0
    output_batch = 1 if net(x1, x2, w1_batch, w2_batch, theta_batch) > 0 else 0
    print(f'Input: ({x1}, {x2}), Online training prediction: {output_online}, Batch training prediction: {output_batch}'
          f', Target: {target_value}')'''
