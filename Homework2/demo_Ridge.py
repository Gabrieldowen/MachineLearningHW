import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
# load data
data = np.loadtxt('crime.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# set training number and testing number
num_train = int(0.1*n)
num_test = int(0.25*n)

# split data into training set and testing set
sample_train = data[0:num_train, 0:-1]
sample_test = data[n-num_test:, 0:-1]
label_train = data[0:num_train, -1]
label_test = data[n-num_test:, -1]

# hyper-parameter
lamda = 0

# TRAIN model and use it to make prediction
model = linear_model.Ridge(lamda)
model.fit(sample_train, label_train)

# PREDICT
label_train_pred = model.predict(sample_train)
label_test_pred = model.predict(sample_test)

# TEST evaluate model error
error_train = mean_squared_error(label_train, label_train_pred)
error_test = mean_squared_error(label_test, label_test_pred)

# print results or draw a figure
print("Training Error = %.4f" % error_train)
print('Testing Error = %.4f' % error_test)
# model.coef_
