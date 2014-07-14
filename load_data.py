import pandas as pd


def load_data(file, loader = pd.read_csv):
    return loader(file)


def load_train_test(train_file, test_file, loader = pd.read_csv):
    return load_data(train_file, loader), load_data(test_file, loader)


def load_train_features(train_file, target_col, transformer, loader = pd.read_csv):
    data = load_data(train_file, loader)
    X_train = transformer.fit_transform(data)
    return X_train, data[target_col]


def load_test_features(test_file, transformer, loader = pd.read_csv):
    data = load_data(test_file, loader)
    return transformer.transform(data)


def load_train_test_features(train_file, test_file, transformer,
                             concat = False,
                             loader = pd.read_csv):
    if concat:
        train, test = load_train_test(train_file, test_file, loader)
        data = pd.concat([train, test])
        X = transformer.fit_transform(data)
        return X[:len(data)], X[len(data):]

    else:
        X_train = load_train_features(train_file, transformer, loader)
        X_test = load_test_features(test_file, transformer, loader)
        return X_train, X_test
