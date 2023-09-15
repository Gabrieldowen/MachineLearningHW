import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# TASK 2

# diabetes dataset
data = np.loadtxt('crime.csv', delimiter=',', skiprows=1)
[n, p] = np.shape(data)


#weights =  np.ones(num_train)
# percentages used for training and testing respectively
num_train = int(0.75*n)
num_test = n - num_train


# split data into training set and testing set
sample_train = data[0:num_train,0:-1]
sample_test = data[num_train:,0:-1]
label_train = data[0:num_train,-1]
label_test = data[num_train:,-1]

# hyper-parameters of your model
alpha = 0.5

# train model and use it to make prediction
# this is often the part you need to implement from scratch
model = linear_model.Ridge(alpha)
model.fit(sample_train, label_train)
label_train_pred = model.predict(sample_train)
label_test_pred = model.predict(sample_test)

# evaluate model error
# this is also the part you need to implement from scratch
error_train = mean_squared_error(label_train, label_train_pred, squared=False)
error_test = mean_squared_error(label_test, label_test_pred, squared=False)

# print results or draw a figure
print(error_train)
print(error_test)
