print('Importing Matplotlib')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

import math
import random

print('Importing Numpy')
import numpy as np

print('Importing PyNeurGen')
from pyneurgen.neuralnet import NeuralNet
from pyneurgen.recurrent import NARXRecurrent
from pyneurgen.nodes import BiasNode, Connection

print('Importing finished')


class PatternGenerator(object):
    def __init__(self, n_samples, debug=False):
        self.debug = debug
        self.n_samples = n_samples
        X = [i/10.0 for i in range(400)]
        # y = np.log(1 + np.cos(X + np.sin(X) * np.sin(X)))
        # y = [math.sin(10*i) for i in X]
        y = [math.log(1 + math.cos(i + math.sin(i) * math.sin(i))) for i in X]
        y_norm = [(i - min(y))/(max(y) - min(y)) for i in y]
        y = y_norm
        self.base_X = X
        self.base_y = y
        Z = [(i,j) for i,j in zip(X, y)]
        # np.random.shuffle(Z)
        random.shuffle(Z)
        X = []
        y = []
        for i, j in Z:
            X.append([i])
            y.append([j])
        
        self.X = X
        self.y = y
        if self.debug: print("X, y generated")

    def better_plot_data(self, classifiers, labels):
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        ax1.scatter(self.X, self.y, label='curva real', alpha=0.5)
        for clf, lbl in zip(classifiers, labels):
            x_test, y_real, y_pred = clf.get_test_values()
            ax2.plot(self.base_X, self.base_y)
            ax2.scatter(x_test, y_real, label=lbl)
            ax2.scatter(x_test, y_pred, label=lbl+' (previsão)')
        ax1.legend()
        plt.show()


class NARXNetwork(object):
    def __init__(self, X, y, hidden_layer_sizes=(10,), debug=False):
        self.X = X
        self.y = y

        in_nodes = 1
        out_nodes = 1
        output_order = 3
        incoming_weight_from_output = .2
        input_order = 1
        incoming_weight_from_input = .4

        self.net = NeuralNet()
        self.net.init_layers(in_nodes, hidden_layer_sizes, out_nodes,
                             NARXRecurrent(output_order, incoming_weight_from_output,
                                 input_order, incoming_weight_from_input) )

        self.net.randomize_network()
        self.net.set_halt_on_extremes(True)
        self.net.set_random_constraint(.5)
        self.net.set_learnrate(.1)

        self.net.set_all_inputs(self.X)
        self.net.set_all_targets(self.y)

        learn_end_point = int(len(self.X) * .8)
        self.net.set_learn_range(0, learn_end_point)
        self.net.set_test_range(learn_end_point + 1, len(self.X) - 1)
        self.net.layers[1].set_activation_type('tanh')

    def fit(self):
        self.net.learn(epochs=100, show_epoch_results=True, random_testing=False)

    def accuracy(self):
        return self.net.test()

    def get_test_values(self):
        # net.get_test_data returns a tuple of input and expected test_data
        # net.test_targets_activations returns a tuple of expected y and predicted y
        test_positions = [item[0][0] for item in self.net.get_test_data()]
        all_targets = [item[1][0] for item in self.net.get_test_data()]
        all_actuals = [item[1][0] for item in self.net.test_targets_activations]
        return test_positions, all_targets, all_actuals


def main():
    print("Começou")
    debug = True
    pg = PatternGenerator(n_samples=200, debug=debug)

    narx = NARXNetwork(pg.X, pg.y, [10,10,])
    narx.fit()
    print(narx.accuracy())

    # pg.better_plot_data([nn for nn in neural_networks], labels)
    pg.better_plot_data([narx], ['NARX'])
    # pg.better_plot_data([], [])

if __name__ == '__main__':
    main()


# References
# https://stackoverflow.com/questions/42713276/valueerror-unknown-label-type-while-implementing-mlpclassifier
# http://www.machinelearningtutorial.net/2017/01/28/python-scikit-simple-function-approximation/
# https://stackoverflow.com/questions/41308662/how-to-tune-a-mlpregressor
# Prediction error http://www.scikit-yb.org/en/latest/api/regressor/peplot.html
# https://stackoverflow.com/questions/41069905/trouble-fitting-simple-data-with-mlpregressor
# https://www.mathworks.com/help/deeplearning/ug/design-time-series-narx-feedback-neural-networks.html;jsessionid=8a2764835d0ef280e4116851f951