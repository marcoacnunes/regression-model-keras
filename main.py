import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

# Download data
concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
data = concrete_data[concrete_data.columns[concrete_data.columns != 'Strength']]
target = concrete_data['Strength']

# A. Baseline Model
def baseline_model():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(data.shape[1],)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

mse_A = []
for _ in range(50):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    model = baseline_model()
    model.fit(X_train, y_train, epochs=50, verbose=0)
    y_pred = model.predict(X_test)
    mse_A.append(mean_squared_error(y_test, y_pred))

# B. Normalize the data
data_norm = (data - data.mean()) / data.std()
mse_B = []
for _ in range(50):
    X_train, X_test, y_train, y_test = train_test_split(data_norm, target, test_size=0.3)
    model = baseline_model()
    model.fit(X_train, y_train, epochs=50, verbose=0)
    y_pred = model.predict(X_test)
    mse_B.append(mean_squared_error(y_test, y_pred))

# C. Increase the number of epochs
mse_C = []
for _ in range(50):
    X_train, X_test, y_train, y_test = train_test_split(data_norm, target, test_size=0.3)
    model = baseline_model()
    model.fit(X_train, y_train, epochs=100, verbose=0)
    y_pred = model.predict(X_test)
    mse_C.append(mean_squared_error(y_test, y_pred))

# D. Increase the number of hidden layers
def deep_model():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(data.shape[1],)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

mse_D = []
for _ in range(50):
    X_train, X_test, y_train, y_test = train_test_split(data_norm, target, test_size=0.3)
    model = deep_model()
    model.fit(X_train, y_train, epochs=50, verbose=0)
    y_pred = model.predict(X_test)
    mse_D.append(mean_squared_error(y_test, y_pred))

print("A. MSE Mean:", np.mean(mse_A), "MSE Std:", np.std(mse_A))
print("B. MSE Mean:", np.mean(mse_B), "MSE Std:", np.std(mse_B))
print("C. MSE Mean:", np.mean(mse_C), "MSE Std:", np.std(mse_C))
print("D. MSE Mean:", np.mean(mse_D), "MSE Std:", np.std(mse_D))
