#!/usr/bin/python3

"""
Logistic Regression based on UCI Vertebral Column Data Set
available at:

    https://archive.ics.uci.edu/ml/datasets/Vertebral+Column

"""

import matplotlib.pyplot as plt
import numpy as np
import random as rnd

SEED = 91

class LogisticRegression():
    ''' Implements logistic regression model '''

    def __init__(self):
        self._sigmoid = lambda x : 1/(1 + np.e**(-x))
        self._fitted = False

    def fit(self, X, y, rate=0.01, epochs=5000):
        self.W, self.b = np.zeros((len(X[0]), 1)), np.zeros((1,1))
        m = len(y)
        for epoch in range(epochs):
            proba = self._sigmoid(np.matmul(X, self.W) + self.b)
            change = proba - y
            dw, db = np.matmul(X.T, change)/m, np.sum(change)
            learn = lambda x, dx : x - rate * dx
            self.W, self.b = learn(self.W, dw), learn(self.b, db)
        self._fitted = True

    def predict(self, X):
        if not self._fitted:
            self.W, self.b = np.zeros((len(X), 1)), np.zeros((1,1))
        return ((self._sigmoid(np.matmul(X, self.W) + self.b)) > 0.5).astype(np.int)

    def accuracy(self, X, y):
        successes = sum([self.predict(X[i]) == y[i] for i in range(len(y))])
        return ((successes/len(y))*100).flat[0]

    def print_model(self):
        if not self._fitted:
            print("Logistic Regression is not trained!")
            return
        print("b:\t{}".format(self.b.flat[0]))
        for i in range(len(self.W)):
            print("w{0}:\t{1}".format(i, self.W.flat[i]))


def load_data(filename, separator=' ', ratio=0.7, create_files=True):
    ''' Returns training and test data from a given file '''
    def split_data(data):
        data = [x.strip().split(separator) for x in data]
        return {'data': np.array([[float(i) for i in x[:-1]] for x in data]), 
                'label': np.array([[0.0 if x[-1] == 'NO' else 1.0] for x in data])}

    def save_data(filename, data):
        with open(filename, 'w') as f:
            f.writelines(data)
        
    with open(filename, 'r') as f:
        lines = f.readlines()
        # randomize the lines
        rnd.seed(SEED)
        rnd.shuffle(lines)
        limit = int(len(lines)*ratio)
        training, test = lines[:limit], lines[limit:]

        if create_files:
            save_data('training.csv', training)
            save_data('test.csv', test)

        return split_data(training), split_data(test)



train, test = load_data("column_3C.dat")
logreg = LogisticRegression()
logreg.fit(train['data'], train['label'], rate=0.01, epochs=50000)
logreg.print_model()
print("\nAccuracy: %.2f"%(logreg.accuracy(test['data'], test['label'])))
