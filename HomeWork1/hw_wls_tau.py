import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
data = np.loadtxt('HomeWork1/crime.csv', delimiter=',', skiprows=1)


def WLS(tau):
    [n, p] = np.shape(data)

    # Split the data
    num_train = int(tau * n)
    num_test = n - num_train

    weights = np.ones(num_train)

    sample_train = data[:num_train, :-1]
    sample_test = data[num_train:, :-1]
    label_train = data[:num_train, -1]
    label_test = data[num_train:, -1]

    # Define your hyperparameter (alpha in Ridge regression)
    alpha = 0.5

    # Implement Weighted Least Squares (WLS) from scratch
    # You need to calculate beta (the model coefficients) using the weights

    X = np.hstack((sample_train, np.ones((num_train, 1))))  # Add a column of ones for the intercept
    W = np.diag(weights)
    X_transpose = np.transpose(X)

    try:
        beta = np.linalg.inv(X_transpose @ W @ X + alpha * np.identity(p)) @ X_transpose @ W @ label_train
    except np.linalg.LinAlgError:
        print(f"Matrix inversion failed for tau={tau}.")
        return

    # Calculate predictions
    sample_train_with_intercept = np.hstack((sample_train, np.ones((num_train, 1))))
    label_train_pred = sample_train_with_intercept @ beta

    sample_test_with_intercept = np.hstack((sample_test, np.ones((num_test, 1))))
    label_test_pred = sample_test_with_intercept @ beta

    # Calculate the error with weighted squared differences and print
    weighted_error_train = np.sqrt(np.sum(weights * (label_train - label_train_pred) ** 2) / num_train)
    print(f"Weighted Training Error (tau={tau}):", weighted_error_train)

    if tau != 1:
        # 2 norm
        weighted_error_test = np.sqrt(np.sum((label_test - label_test_pred) ** 2) / num_test)
        return weighted_error_train, weighted_error_test
    else:
        return weighted_error_train, 0

    print()


# Pick 5 values of τ, including τ = 1
tau_values = [0.1, 0.25, 0.33, 0.5, 1]

# Calculate errors for each τ value
training_errors = []
testing_errors = []

for tau in tau_values:
    train_err, test_err = WLS(tau)
    training_errors.append(train_err)
    if test_err is not None:
        testing_errors.append(test_err)
    else:
        testing_errors.append(0)  # Set testing error to 0 when tau is 1

# Create a graph to report both training and testing errors versus τ
plt.figure(figsize=(8, 6))
plt.plot(tau_values, training_errors, marker='o', label='Training Error')
plt.plot(tau_values, testing_errors, marker='o', label='Testing Error')
plt.xlabel('Tau (τ)')
plt.ylabel('Error')
plt.title('Errors versus Tau (τ)')
plt.legend()
plt.grid(True)
plt.show()