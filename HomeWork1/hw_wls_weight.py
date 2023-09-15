import numpy as np

# Load your dataset
data = np.loadtxt('crime.csv', delimiter=',', skiprows=1)


def WLS(tau, threshold):
    [n, p] = np.shape(data)

    # Split the data
    num_train = int(tau * n)
    num_test = n - num_train

    weights = np.ones(num_train)

    sample_train = data[:num_train, :-1]
    sample_test = data[num_train:, :-1]
    label_train = data[:num_train, -1]
    label_test = data[num_train:, -1]

    # split data
    high_risk_communities = data[data[:, -1] > threshold]
    low_risk_communities = data[data[:, -1] <= threshold]

    # get length of groups
    num_high_risk = len(high_risk_communities)
    num_low_risk = len(low_risk_communities)


    # weights for risk groups
    wh = 0.01
    wl = 1

    weights[:num_high_risk] = wh  # Assign wh to high-risk group
    weights[num_high_risk:] = wl  # Assign wl to low-risk group

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

    # Error on high-risk group
    high_risk_indices = label_train > threshold
    error_high_risk = np.sqrt(
        np.sum(weights[high_risk_indices] * (label_train[high_risk_indices] - label_train_pred[high_risk_indices]) ** 2) / num_train)
    print(f"high Risk Error:", error_high_risk)
    # Error on low-risk group
    error_low_risk = np.sqrt(
        np.sum(weights[~high_risk_indices] * (label_train[~high_risk_indices] - label_train_pred[~high_risk_indices]) ** 2) / num_train)
    print(f"Low Risk Error:", error_low_risk)



    # Calculate the error with weighted squared differences and print
    weighted_error_train = np.sqrt(np.sum(weights * (label_train - label_train_pred) ** 2) / num_train)
    print(f"Weighted Training Error (tau={tau}):", weighted_error_train)





WLS(1, 0.8)
