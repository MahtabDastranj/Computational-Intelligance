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

