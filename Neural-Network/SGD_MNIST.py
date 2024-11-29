import numpy as np
import matplotlib.pyplot as plt
import time


# Reading The Train Set
def read_train_set(read):
    traini_path = r"E:\AUT\MNIST\train-images-idx3-ubyte\train-images.idx3-ubyte"
    trainl_path = r"E:\AUT\MNIST\train-labels-idx1-ubyte\train-labels.idx1-ubyte"

    with open(traini_path, 'rb') as train_images_file, open(trainl_path, 'rb') as train_labels_file:
        train_images_file.seek(4)
        num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
        train_images_file.seek(16)
        train_labels_file.seek(8)

        train_set = []
        read = min(read, num_of_train_images)  # Adjust read to the maximum available images
        for n in range(read):
            image = np.zeros((784, 1))
            for i in range(784):
                image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256

            label_value = int.from_bytes(train_labels_file.read(1), 'big')
            label = np.zeros((10, 1))
            label[label_value, 0] = 1

            train_set.append((image, label))
    return train_set


# Reading The Test Set
def read_test_set(read):
    ti_path = r"E:\AUT\MNIST\t10k-images-idx3-ubyte\t10k-images.idx3-ubyte"
    tl_path = r"E:\AUT\MNIST\t10k-labels-idx1-ubyte\t10k-labels.idx1-ubyte"

    with open(ti_path, 'rb') as test_images_file, open(tl_path, 'rb') as test_labels_file:
        test_images_file.seek(4)
        num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
        test_images_file.seek(16)
        test_labels_file.seek(8)

        test_set = []
        read = min(read, num_of_test_images)  # Adjust read to the maximum available images
        for n in range(read):
            image = np.zeros((784, 1))
            for i in range(784):
                image[i, 0] = int.from_bytes(test_images_file.read(1), 'big') / 256

            label_value = int.from_bytes(test_labels_file.read(1), 'big')
            label = np.zeros((10, 1))
            label[label_value, 0] = 1

            test_set.append((image, label))
    return test_set


def plot_unique_images_with_labels(dataset):
    """
    Plot a grid of images, each showing a unique digit (0-9).
    :param dataset: The dataset containing image-label pairs.
    """
    # Dictionary to store the first occurrence of each digit
    unique_digits = {}

    # Loop through the dataset and collect one example per digit
    for image, label in dataset:
        label_value = np.argmax(label)  # Decode the one-hot encoded label
        if label_value not in unique_digits:
            unique_digits[label_value] = image  # Store the first occurrence of the digit

        # Stop if we have all digits (0-9)
        if len(unique_digits) == 10:
            break

    # Plot the collected unique digits
    plt.figure(figsize=(10, 5))
    for i in range(10):
        image = unique_digits[i]
        image_reshaped = image.reshape((28, 28))  # Reshape 784x1 to 28x28

        plt.subplot(2, 5, i + 1)
        plt.imshow(image_reshaped, cmap='gray')
        plt.title(f"Label: {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


'Checking whether the files are correctly read'
# Testing the function
train_set = read_train_set(20)  # Read 20 training images
plot_unique_images_with_labels(train_set)

'Multi Layer Perceptron (MLP)'


def sigmoid(x): return 1 / (1 + np.exp(-x))


def initialization(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.normal(0, 1, (hidden_size, input_size))
    W2 = np.random.normal(0, 1, (hidden_size, hidden_size))
    W3 = np.random.normal(0, 1, (output_size, hidden_size))
    b1, b2, b3 = np.zeros((hidden_size, 1)), np.zeros((hidden_size, 1)), np.zeros((output_size, 1))

    return W1, W2, W3, b1, b2, b3


def forward_propagation(data, W1, W2, W3, b1, b2, b3):
    net1 = W1 @ data + b1
    act1 = sigmoid(net1)
    net2 = W2 @ act1 + b2
    act2 = sigmoid(net2)
    net3 = W3 @ act2 + b3
    act3 = sigmoid(net3)
    return net1, act1, net2, act2, net3, act3


def selection(act3): return np.argmax(act3, axis=0)


def accuracy(predictions, true_labels): return np.sum(predictions == true_labels) / len(true_labels)


def evaluate(train_set):
    input_size = 784
    hidden_size = 16
    output_size = 10

    # Initialize weights and biases
    W1, W2, W3, b1, b2, b3 = initialization(input_size, hidden_size, output_size)

    # Extract data and labels
    data = np.hstack([x[0] for x in train_set])  # Images as column vectors
    labels = np.hstack([x[1] for x in train_set])  # Labels as column vectors

    start_time = time.time()

    # Perform forward propagation
    _, _, _, _, _, output = forward_propagation(data, W1, W2, W3, b1, b2, b3)

    # Convert output to predictions
    predictions = selection(output)
    true_labels = np.argmax(labels, axis=0)

    # Calculate accuracy
    precision = accuracy(predictions, true_labels)

    duration = time.time() - start_time

    print(f"Accuracy: {precision * 100:.2f}%")
    print(f"Duration: {duration:.6f} seconds")

    return precision, duration


train_set = read_train_set(600)  # Read the first 600 images from the training set
precision, duration = evaluate(train_set)  # Evaluate the network

'Training the MLP by implementing Backpropagation: Stochastic Gradient Descent(SGD)'


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Backward propagation
def back_propagation(X, Y, net1, act1, net2, act2, net3, act3, W1, W2, W3, lr, b1, b2, b3):
    # Output layer error
    dZ3 = act3 - Y  # [10 x batch_size]
    dW3 = dZ3 @ act2.T  # [10 x 16]
    db3 = np.sum(dZ3, axis=1, keepdims=True)  # [10 x 1]

    # Hidden layer 2 error
    dZ2 = (W3.T @ dZ3) * sigmoid_derivative(net2)  # [16 x batch_size]
    dW2 = dZ2 @ act1.T  # [16 x 16]
    db2 = np.sum(dZ2, axis=1, keepdims=True)  # [16 x 1]

    # Hidden layer 1 error
    dZ1 = (W2.T @ dZ2) * sigmoid_derivative(net1)  # [16 x batch_size]
    dW1 = dZ1 @ X.T  # [16 x 784]
    db1 = np.sum(dZ1, axis=1, keepdims=True)  # [16 x 1]

    # Update weights and biases
    W3 -= lr * dW3
    b3 -= lr * db3
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    return W1, b1, W2, b2, W3, b3


# Loss function with batch normalization
def mle(Y, A3):
    return 0.5 * np.sum((Y - A3) ** 2) / Y.shape[1]


# SGD with Backpropagation
def train_sgd(train_set, input_size, hidden_size, output_size, batch_size, lr, epochs):
    # Initialize weights and biases
    W1, W2, W3, b1, b2, b3 = initialization(input_size, hidden_size, output_size)

    # Extract data and labels
    data = np.hstack([x[0] for x in train_set])  # Images as column vectors
    labels = np.hstack([x[1] for x in train_set])  # Labels as column vectors

    # Track loss
    loss_values = []

    start_time = time.time()

    for epoch in range(epochs):
        # Shuffle the training set
        perm = np.random.permutation(data.shape[1])
        data, labels = data[:, perm], labels[:, perm]

        for i in range(0, data.shape[1], batch_size):
            X_batch = data[:, i:i + batch_size]
            Y_batch = labels[:, i:i + batch_size]

            # Forward pass
            net1, act1, net2, act2, net3, act3 = forward_propagation(X_batch, W1, W2, W3, b1, b2, b3)

            # Backward pass
            W1, b1, W2, b2, W3, b3 = back_propagation(
                X_batch, Y_batch, net1, act1, net2, act2, net3, act3, W1, W2, W3, lr, b1, b2, b3
            )

        # Compute loss at the end of each epoch
        _, _, _, _, _, A3_full = forward_propagation(data, W1, W2, W3, b1, b2, b3)
        loss = mle(labels, A3_full)
        loss_values.append(loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds.")

    return W1, b1, W2, b2, W3, b3, loss_values, duration


# Evaluate Function
def sgd_evaluate(data, labels, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_propagation(data, W1, W2, W3, b1, b2, b3)
    predictions = np.argmax(A3, axis=0)
    true_labels = np.argmax(labels, axis=0)
    return accuracy(predictions, true_labels)


# Main Script
train_set = read_train_set(600)  # Load the first 600 images
data = np.hstack([x[0] for x in train_set])  # Input data
labels = np.hstack([x[1] for x in train_set])  # One-hot encoded labels

# Hyperparameters
input_size = 784
hidden_size = 16
output_size = 10
batch_size = 6
lr = 1
epochs = 100

# Train the network
W1, b1, W2, b2, W3, b3, loss_values, duration = train_sgd(
    train_set, input_size, hidden_size, output_size, batch_size, lr, epochs
)

# Evaluate accuracy
acc = sgd_evaluate(data, labels, W1, b1, W2, b2, W3, b3)
print(f"Accuracy: {acc * 100:.2f}%")

# Plot loss over epochs
plt.plot(range(1, epochs + 1), loss_values, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()
