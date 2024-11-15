import numpy as np
import time
import matplotlib.pyplot as plt

'Q N0. 1'
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([1, 0, 0, 0])

lr = 0.1  # learning rate
print(np.random.random())  # Generate a random number between 0 (inclusive) and 1 (exclusive)
def online_training(inputs, target, lr):
    w1 = np.random.random()
    w2 = np.random.random()
    theta = np.random.random()
    epoch = 1
    sigma = 0
    e1, e2 = None, None

    while e1 and e2 not in [0, 0]:
        for input in inputs:
            sigma = (input[0] * w1) + (input[1] * w2)
            if sigma > theta: output = 1, else: output = 0
