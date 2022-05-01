import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import math


"""
## 2.a
"""


def load_data(fake_text, real_text):
    label = np.array([0] * len(fake_text) + [1] * len(real_text)).reshape(-1, 1)
    corpus = fake_text + real_text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    tr_corpus, va_corpus, tr_label, va_label = train_test_split(X.toarray(), label, train_size=0.7, shuffle=True)
    va_corpus, te_corpus, va_label, te_label = train_test_split(va_corpus, va_label, train_size=0.5, shuffle=True)
    col_names = vectorizer.get_feature_names() + ["target_y"]
    train = pd.DataFrame(np.concatenate((tr_corpus, tr_label), axis=1), columns=col_names)
    valid = pd.DataFrame(np.concatenate((va_corpus, va_label), axis=1), columns=col_names)
    test = pd.DataFrame(np.concatenate((te_corpus, te_label), axis=1), columns=col_names)
    return [train, valid, test]


"""
## 2.b
"""


def select_model(train, valid, max_depth_list, criterion_list):
    trainY = train["target_y"].values
    trainX = train.drop(["target_y"], axis=1).values
    validY = valid["target_y"].values
    validX = valid.drop(["target_y"], axis=1).values
    keys = ["max_depth", "criterion", "accuray"]
    result_dict = {k: [] for k in keys}
    best_model = None
    best_acc = 0
    for max_depth in max_depth_list:
        for criterion in criterion_list:
            model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
            model.fit(trainX, trainY)
            valid_pred = model.predict(validX)
            metric = accuracy_score(validY, valid_pred)
            print(f" max_depth : {max_depth} , criterion : {criterion} , accuray : {metric*100}")
            result_dict["max_depth"] = max_depth
            result_dict["criterion"] = criterion
            result_dict["accuray"] = metric
            if best_acc < metric:
                best_model = model
                best_acc = metric
    return best_model


"""
## 2.d
"""


def calc_entropy(column):
    counts = np.bincount(column)
    probabilities = counts / len(column)
    entropy = 0
    for prob in probabilities:
        if prob > 0:
            entropy += prob * math.log(prob, 2)
    return -entropy


def compute_information_gain(data, split_name, target_name):
    original_entropy = calc_entropy(data[target_name])
    values = data[split_name].unique()
    left_split = data[data[split_name] == values[0]]
    right_split = data[data[split_name] == values[1]]
    to_subtract = 0
    for subset in [left_split, right_split]:
        prob = subset.shape[0] / data.shape[0]
        to_subtract += prob * calc_entropy(subset[target_name])
    return original_entropy - to_subtract
