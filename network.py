import gzip
import random
import json
import os.path

import numpy as np

class NeuralNetwork(object):

    def __init__(self):
        self._save_file = 'ann_wb.json'
        self._sigmoid = lambda x : 1.0/(1.0+np.exp(-x))

        if not self._load_neural_network():
            layers = [6, 10, 10, 3]
            self._n_layers = len(layers)
            self._biases = [np.random.randn(y, 1) for y in layers[1:]]
            self._weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
            self._sigprime = lambda x : self._sigmoid(x)*(1-self._sigmoid(x))

            training, self._tests = self._load_data("column_3C.dat", create_files=False)
            self._train(training, batch_size=10, epochs=500, eta=0.15)
            self._save_neural_network()
            print("Training complete!")
        else:
            print("Neural Network Loaded!")

        print("Accuracy of {} %".format(self._accuracy))


    def get_digit(self, inputs):
        ''' Evaluates a set of pixels 784*1 and returns predicted digit. Feedforward for evaluation'''
        for w, b in zip(self._weights, self._biases):
            inputs = self._sigmoid(np.dot(w, inputs)+b)
        return np.argmax(inputs)


    def get_accuracy(self):
        ''' returns accuracy percentage '''
        results = [(self.get_digit(pixels), digit) for (pixels, digit) in self._tests]
        passed = sum(1 for (prediction, digit) in results if prediction == digit)
        return (passed / len(self._tests)) * 100


    def _load_neural_network(self):
        if not os.path.exists(self._save_file):
            return False

        with open(self._save_file, 'r', encoding='utf-8') as f:
            x = json.loads(f.read())
            self._accuracy = x['accuracy']
            self._biases = np.array(x['biases'])
            self._weights = np.array(x['weights'])

        return True


    def _save_neural_network(self):
        b = [b.astype('float64').tolist() for b in self._biases]
        w = [w.astype('float64').tolist() for w in self._weights]
        json.dump({'weights':w, 'biases':b, 'accuracy':self._accuracy}, open(self._save_file, 'w', encoding='utf-8'))


    def _load_data(self, filename, separator=' ', ratio=0.7, create_files=True):
        ''' load vertebral column data, return training data and test data '''
        import random as rnd
        SEED = 91

        def _get_value(x):
            if x == 'NO':
                return 0
            elif x == 'DH':
                return 1
            elif x == 'SL':
                return 2
            # this should not happen
            return 3

        def _to_vector(x):
            e = np.zeros((3, 1))
            e[_get_value(x)] = 1.0
            return e

        def split_train_data(data):
            data = [x.strip().split(separator) for x in data]
            return list(zip(np.array([np.reshape([float(i)/10.0 for i in x[:-1]], (6, 1)) for x in data], dtype=np.float128), 
                    np.array([_to_vector(x[-1]) for x in data], dtype=np.float128)))

        def split_test_data(data):
            data = [x.strip().split(separator) for x in data]
            return list(zip(np.array([np.reshape([float(i)/10.0 for i in x[:-1]], (6, 1)) for x in data], dtype=np.float128), 
                    np.array([_get_value(x[-1]) for x in data], dtype=np.float128)))

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

            return split_train_data(training), split_test_data(test)


    def _train(self, data, batch_size=10, epochs=30, eta=3.0):
        ''' Trains neural network with Stochastic Gradient Descent '''
        import click
        with click.progressbar(label="Training", length=len(data)*epochs) as bar:
            for _ in range(epochs):
                random.shuffle(data)
                batches = [data[x:x+batch_size] for x in range(0, len(data), batch_size)]
                for batch in batches:
                    self._update_batch(batch, eta)
                    bar.update(batch_size)
        self._accuracy = round(self.get_accuracy(),2)


    def _update_batch(self, batch, eta):
        change_b = [np.zeros(b.shape) for b in self._biases]
        change_w = [np.zeros(w.shape) for w in self._weights]
        for pixels, correct_response in batch:
            batch_change_b, batch_change_w = self._backpropagation(pixels, correct_response)
            change_b = [cb+bcb for cb, bcb in zip(change_b, batch_change_b)]
            change_w = [cw+bcw for cw, bcw in zip(change_w, batch_change_w)]
        self._weights = [w-eta*(cw/len(batch)) for w, cw in zip(self._weights, change_w)]
        self._biases = [b-eta*(cb/len(batch)) for b, cb in zip(self._biases, change_b)]


    def _backpropagation(self, x, y):
        # feedforward for training!
        activations, zvectors = [x], []
        for b, w in zip(self._biases, self._weights):
            zvectors.append(np.dot(w, activations[-1])+b)
            activations.append(self._sigmoid(zvectors[-1]))

        change_b = [np.zeros(b.shape) for b in self._biases]
        change_w = [np.zeros(w.shape) for w in self._weights]
        delta = (activations[-1] - y) * self._sigprime(zvectors[-1])
        change_b[-1] = delta
        change_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self._n_layers):
            delta = np.dot(self._weights[-l+1].transpose(), delta) * self._sigprime(zvectors[-l])
            change_b[-l] = delta
            change_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (change_b, change_w)


NeuralNetwork()
