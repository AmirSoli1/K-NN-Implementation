from nearest_neighbour import learnknn, predictknn, gensmallm
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

# Function to calculate k-nn error
def knn_error(k, x_train, y_train, x_test, y_test):
    classifier = learnknn(k, x_train, y_train)
    y_pred = predictknn(classifier, x_test)
    error = np.mean(np.vstack(y_test) != np.vstack(y_pred))
    return error

# Function to generate data and calculate average error
def run_experiment(k, sample_size, num_trials):
    data = np.load('mnist_all.npz')
    errors = []
    for _ in range(num_trials):
        # Generate random training and test samples
        train2 = data['train2']
        train3 = data['train3']
        train5 = data['train5']
        train6 = data['train6']

        test2 = data['test2']
        test3 = data['test3']
        test5 = data['test5']
        test6 = data['test6']

        x_train, y_train = gensmallm([train2, train3, train5, train6], [2, 3, 5, 6], sample_size)

        x_test, y_test = gensmallm([test2, test3, test5, test6], [2, 3, 5, 6], len(test2) + len(test3) + len(test5) + len(test6))

        
        # Calculate error
        error = knn_error(k, x_train, y_train, x_test, y_test)
        errors.append(error)

    # Calculate average error and error bars
    avg_error = np.mean(errors)
    min_error = np.min(errors)
    max_error = np.max(errors)

    return avg_error, min_error, max_error

# Experiment parameters
sample_sizes = np.arange(1, 101, 5)
num_trials = 10

# Run experiments
avg_errors = []
min_errors = []
max_errors = []

for size in sample_sizes:
    avg_error, min_error, max_error = run_experiment(1,size, num_trials)
    avg_errors.append(avg_error)
    min_errors.append(min_error)
    max_errors.append(max_error)

avg_errors, min_errors, max_errors = np.array(avg_errors), np.array(min_errors), np.array(max_errors)

# Plot results
plt.errorbar(sample_sizes, avg_errors, yerr=[avg_errors - min_errors, max_errors - avg_errors], fmt='o-')
plt.xlabel('Training Sample Size')
plt.ylabel('Average Test Error')
plt.title('k-NN Performance with Varying Training Sample Size')
plt.legend()
plt.show()

# Values of k to be tested
k_values = np.arange(1, 12)

# Run experiments for different values of k
avg_errors = []
min_errors = []
max_errors = []

for k in k_values:
    avg_error, min_error, max_error = run_experiment(k, 200, num_trials)
    avg_errors.append(avg_error)
    min_errors.append(min_error)
    max_errors.append(max_error)

# Plot results
plt.plot(k_values, avg_errors, marker='o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Average Test Error')
plt.title('k-NN Performance with Varying k (Sample Size = 200)')
plt.show()

# Find the optimal value of k (minimum average test error)
optimal_k = k_values[np.argmin(avg_errors)]
print(f"Optimal value of k: {optimal_k}")
