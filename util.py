import numpy as np
import sklearn.preprocessing as prep
import math

def standard_scaler(X_train, X_test):
    train_samples, train_nx, train_ny = X_train.shape
    test_samples, test_nx, test_ny = X_test.shape
    
    X_train = X_train.reshape((train_samples, train_nx * train_ny))
    X_test = X_test.reshape((test_samples, test_nx * test_ny))
    
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    X_train = X_train.reshape((train_samples, train_nx, train_ny))
    X_test = X_test.reshape((test_samples, test_nx, test_ny))
    
    return X_train, X_test


def preprocess_data(etc_data, seq_len,number_features):
    result = []
    seq_len +=1
    for index in range(len(etc_data) - seq_len):
        result.append(etc_data[index : index + seq_len])
    result = np.array(result)
    row = round(0.9 * result.shape[0])
    
    train = result[: int(row), :]
    
    train, result = standard_scaler(train, result)
    
    X_train = train[:, : -1]
    y_train = train[:, -1][: ,-2]
    X_test = result[int(row) :, : -1]
    y_test = result[int(row) :, -1][ : ,-2]


    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], number_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], number_features))  

    return [X_train, y_train, X_test, y_test]

def score(x_test, y_test, model):
    Score = model.evaluate(x_test, y_test, verbose=0)
    print('Score: %.2f MSE (%.2f RMSE)' % (Score[0], math.sqrt(Score[0])))

def predict(x_test, model):
    return model.predict(x_test)
