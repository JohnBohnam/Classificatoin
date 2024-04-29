import pandas as pd
from sklearn.model_selection import train_test_split

datapath = 'data/rp.data'


def get_data():
    data = pd.read_csv(datapath, header=None)
    data.drop(columns=[0], inplace=True)

    data[10] -= 2
    data[10] //= 2

    name_mapping = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'y'}
    data.rename(columns=name_mapping, inplace=True)
    for col in data.columns:
        if col == 'y':
            continue
        data[col] -= 1

    return data


def get_X_y(data):
    X = data.drop(columns=['y']).to_numpy()
    y = data['y'].to_numpy()
    return X, y


def get_train_test_data(data):
    X, y = get_X_y(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333)
    return X_train, X_test, y_train, y_test