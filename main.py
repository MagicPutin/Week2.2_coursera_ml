import sklearn.linear_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

# classification algorithm
clf = sklearn.linear_model.Perceptron(random_state=241, max_iter=5, tol=None)

# data
test = pd.read_csv('data/perceptron-test.csv', names=['1', '2', '3'])
train = pd.read_csv('data/perceptron-train.csv', names=['1', '2', '3'])
# converting to np.array
train_data = train.loc[:, '2':'3'].to_numpy()
test_data = test.loc[:, '2':'3'].to_numpy()

# without scaling data
clf.fit(train_data, train['1'].to_numpy())
score_without_scaling = sklearn.metrics.accuracy_score(test['1'].to_numpy(), clf.predict(test_data))

print(score_without_scaling)

# with scaling data
scaler = sklearn.preprocessing.StandardScaler()

train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

clf.fit(train_data_scaled, train['1'].to_numpy())
score_with_scaling = sklearn.metrics.accuracy_score(test['1'].to_numpy(), clf.predict(test_data_scaled))
print(score_with_scaling)

with open('answers/task.txt', 'w') as task:
    task.write(str(round(score_with_scaling-score_without_scaling, 3)))
