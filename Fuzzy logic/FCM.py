import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# FCM
def calculate_membership_matrix(data, centers, m):
    n_point = data.shape[0]
    n_cluster = centers.shape[0]
    membership_matrix = np.zeros((n_point, n_cluster))

    for k in range(n_point):
        for i in range(n_cluster):
            numerator = np.linalg.norm(data[k] - centers[i])
            denominator = np.sum([np.linalg.norm(data[k] - centers[j]) ** (2 / (m - 1)) for j in range(n_cluster)])
            if denominator == 0 or numerator == 0:
                membership_matrix[k, i] = 1
            else:
                membership_matrix[k, i] = (numerator / denominator) ** -1
    return membership_matrix


def calculate_cluster_centers(data, membership_matrix, m):
    num_clusters = membership_matrix.shape[1]
    num_features = data.shape[1]
    centers = np.zeros((num_clusters, num_features))

    for i in range(num_clusters):
        numerator = np.sum((membership_matrix[:, i] ** m)[:, None] * data, axis=0)
        denominator = np.sum(membership_matrix[:, i] ** m)
        if denominator == 0:
            centers[i] = numerator
        else:
            centers[i] = numerator / denominator
    return centers


def calculate_cost_function(data, centers, membership_matrix, m):
    cost = 0
    for k in range(data.shape[0]):
        for i in range(centers.shape[0]):
            cost += (membership_matrix[k, i] ** m) * np.linalg.norm(data[k] - centers[i]) ** 2
    return cost


def fcm(data, num_clusters, m=2, max_iter=200, tol=0.5):
    centers = data[np.random.choice(data.shape[0], num_clusters, replace=False)]
    prev_cost = np.inf

    for iteration in range(max_iter):
        membership_matrix = calculate_membership_matrix(data, centers, m)
        centers = calculate_cluster_centers(data, membership_matrix, m)
        cost = calculate_cost_function(data, centers, membership_matrix, m)

        if iteration % 40 == 0:print(f"Iteration {iteration}: Cost = {cost}")

        elif abs(prev_cost - cost) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break
        prev_cost = cost
    else:
        print(f"Couldn't converge after {max_iter} iterations.")

    return centers, membership_matrix, cost


def cost_visualization(data, max_clusters, m=2):
    costs = []
    for c in range(1, max_clusters + 1):
        _, _, cost = fcm(data, c, m)
        costs.append(cost)

    plt.plot(range(1, max_clusters + 1), costs, marker='o', linestyle='--', color='blue')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Cost Function')
    plt.title('Cost Function vs. Number of Clusters')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

    # Elbow method
    suggested_clusters = np.argmin(np.diff(np.diff(costs))) + 2
    print(f"Suggested Number of Clusters: {suggested_clusters}")
    return suggested_clusters


def clustering_visualization(data, num_clusters, m=2):
    centers, membership_matrix, _ = fcm(data, num_clusters, m)
    cluster_labels = np.argmax(membership_matrix, axis=1)
    print(f"Cluster Centers:\n{centers}")

    num_dimensions = data.shape[1]
    if num_dimensions == 2:
        plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='plasma', s=50, alpha=0.7)
        plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='*', s=200, label='Centers')
        plt.title(f'Clustering Result (C={num_clusters})')
        plt.legend()
        plt.show()

    elif num_dimensions == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=cluster_labels, cmap='coolwarm', s=50, alpha=0.7)
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color='black', marker='*', s=200, label='Centers')
        ax.set_title(f'Clustering Result (C={num_clusters})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.show()
    else:
        print("Dimension is too high it can't be processed.")


if __name__ == "__main__":
    data_files = ['data1.csv', 'data2.csv', 'data3.csv', 'data4.csv']
    sample_files = ['data1.csv', 'data2.csv']

    for file in sample_files:
        print(f"Processing file: {file}")

        data = pd.read_csv(file)
        print(data.head())
        data = data.values
        num_dimensions = data.shape[1]
        print(f"Dimensions: {num_dimensions}")

        if num_dimensions == 4:
            # First 3 columns for clustering
            data = data[:, :3]
            num_dimensions = data.shape[1]

        if num_dimensions > 4:
            print("Data has more than 4 dimensions. Cannot demonstrate.")
        else:
            print(f"FCM : {file}")
            offered_clusters = cost_visualization(data, max_clusters=5, m=2)
            clustering_visualization(data, 3, m=2)

