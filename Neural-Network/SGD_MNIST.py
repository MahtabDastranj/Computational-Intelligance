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

'Forward Propagation'


def logistic_activation(x):
    """Applies the logistic sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def initialize_weights(input_size, hidden1_size, hidden2_size, output_size):
    """
    Initialize weight matrices with normal random numbers and biases with zeros.
    """
    weights = {
        'W1': np.random.normal(0, 1, (hidden1_size, input_size)),
        'b1': np.zeros((hidden1_size, 1)),
        'W2': np.random.normal(0, 1, (hidden2_size, hidden1_size)),
        'b2': np.zeros((hidden2_size, 1)),
        'W3': np.random.normal(0, 1, (output_size, hidden2_size)),
        'b3': np.zeros((output_size, 1))
    }
    return weights


def forward_propagation(X, weights):
    """
    Perform forward propagation through the network.
    :param X: Input data (shape: input_size x num_samples)
    :param weights: Dictionary containing weights and biases
    :return: Output of the network
    """
    # Layer 1
    Z1 = np.dot(weights['W1'], X) + weights['b1']
    A1 = logistic_activation(Z1)

    # Layer 2
    Z2 = np.dot(weights['W2'], A1) + weights['b2']
    A2 = logistic_activation(Z2)

    # Output Layer
    Z3 = np.dot(weights['W3'], A2) + weights['b3']
    A3 = logistic_activation(Z3)  # Output probabilities

    return Z1, A1, Z2, A2, Z3, A3


def calculate_accuracy(predictions, labels):
    """
    Calculate accuracy as the percentage of correct predictions.
    :param predictions: Predicted probabilities (output of the network)
    :param labels: True labels (one-hot encoded)
    :return: Accuracy as a percentage
    """
    predicted_labels = np.argmax(predictions, axis=0)
    true_labels = np.argmax(labels, axis=0)
    correct_predictions = np.sum(predicted_labels == true_labels)
    accuracy = (correct_predictions / labels.shape[1]) * 100
    return accuracy


# Main logic
def main():
    # Load the first 600 training images
    train_set = read_train_set(600)
    X = np.hstack([sample[0] for sample in train_set])  # Input matrix (784 x 600)
    Y = np.hstack([sample[1] for sample in train_set])  # Labels matrix (10 x 600)

    # Initialize weights
    input_size = 784
    hidden1_size = 16
    hidden2_size = 16
    output_size = 10
    weights = initialize_weights(input_size, hidden1_size, hidden2_size, output_size)

    # Perform forward propagation and measure runtime
    start_time = time.time()
    _, _, _, _, _, output = forward_propagation(X, weights)
    end_time = time.time()

    # Calculate accuracy
    accuracy = calculate_accuracy(output, Y)

    # Print results
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Time taken: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()

'SGD Implementation'


def logistic_derivative(x):
    return logistic_activation(x) * (1 - logistic_activation(x))


# Error calculation: 0.5 * (d - y)^2
def calculate_error(d, y):
    return 0.5 * np.sum((d - y) ** 2)


# Backward propagation
def backward_propagation(X, Y, Z1, A1, Z2, A2, Z3, A3, weights):
    # Output layer error
    delta3 = (A3 - Y) * logistic_derivative(Z3)
    dW3 = np.dot(delta3, A2.T)
    db3 = np.sum(delta3, axis=1, keepdims=True)

    # Hidden layer 2 error
    delta2 = np.dot(weights['W3'].T, delta3) * logistic_derivative(Z2)
    dW2 = np.dot(delta2, A1.T)
    db2 = np.sum(delta2, axis=1, keepdims=True)

    # Hidden layer 1 error
    delta1 = np.dot(weights['W2'].T, delta2) * logistic_derivative(Z1)
    dW1 = np.dot(delta1, X.T)
    db1 = np.sum(delta1, axis=1, keepdims=True)

    gradients = {'dW3': dW3, 'db3': db3, 'dW2': dW2, 'db2': db2, 'dW1': dW1, 'db1': db1}
    return gradients


# Update weights
def update_weights(weights, gradients, learning_rate):
    for key in weights.keys():
        weights[key] -= learning_rate * gradients['d' + key]
    return weights


# Stochastic Gradient Descent (SGD) Training
def train_network(X, Y, weights, batch_size, learning_rate, epochs):
    num_samples = X.shape[1]
    error_history = []

    start_time = time.time()

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(num_samples)
        X, Y = X[:, indices], Y[:, indices]

        epoch_error = 0

        # Mini-batch training
        for i in range(0, num_samples, batch_size):
            X_batch = X[:, i:i + batch_size]
            Y_batch = Y[:, i:i + batch_size]

            # Forward pass
            Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X_batch, weights)

            # Calculate error for the batch
            batch_error = calculate_error(Y_batch, A3)
            epoch_error += batch_error

            # Backward pass
            gradients = backward_propagation(X_batch, Y_batch, Z1, A1, Z2, A2, Z3, A3, weights)

            # Update weights
            weights = update_weights(weights, gradients, learning_rate)

        # Store epoch error
        error_history.append(epoch_error / num_samples)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Error: {error_history[-1]:.6f}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    return weights, error_history


# Plot error vs epoch
def plot_error(error_history):
    plt.figure(figsize=(10, 6))
    plt.plot(error_history, label="Training Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Error vs. Epoch")
    plt.legend()
    plt.grid()
    plt.show()


# Main function
def m():
    # Load the first 600 samples of the training set
    train_set = read_train_set(600)
    X = np.hstack([sample[0] for sample in train_set])  # Input matrix (784 x 600)
    Y = np.hstack([sample[1] for sample in train_set])  # Labels matrix (10 x 600)

    # Initialize network parameters
    input_size = 784
    hidden1_size = 16
    hidden2_size = 16
    output_size = 10
    weights = initialize_weights(input_size, hidden1_size, hidden2_size, output_size)

    # Train the network
    batch_size = 6
    learning_rate = 1
    epochs = 100
    weights, error_history = train_network(X, Y, weights, batch_size, learning_rate, epochs)

    # Forward pass with final weights
    _, _, _, _, _, A3 = forward_propagation(X, weights)

    # Calculate accuracy
    accuracy = calculate_accuracy(A3, Y)
    print(f"Final Accuracy: {accuracy:.2f}%")

    # Plot error vs epoch
    plot_error(error_history)
    plt.show()


if __name__ == "__m__":
    m()

