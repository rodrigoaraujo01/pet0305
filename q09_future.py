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
        X = [i/10 for i in range(100)]  # i de 0 a 10
        # y = [math.sin(20*i) for i in X]
        y = [math.log(1 + math.cos(i + math.sin(i) * math.sin(i))) for i in X]
        # y = [math.log(1 + math.cos(i + math.sin(i))) for i in X]
        y_norm = [(i - min(y))/(max(y) - min(y)) for i in y]
        y = y_norm
        self.base_X = X
        self.base_y = y
        self.shift = 3
        Z = [(i,j) for i,j in zip(X[:-self.shift], y[self.shift:])]
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
        fig = plt.figure()
        fig.suptitle(labels[0])
        ax1 = plt.subplot(2, 2, 1)
        ax1.set_title('Curva original')
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_title('Valores obtidos')
        ax3 = plt.subplot(2, 2, 3)
        ax3.set_title('MSE por época')
        ax4 = plt.subplot(2, 2, 4)
        ax4.set_title('Erro de predição')
        ax1.scatter(self.X, self.y, label='curva real', alpha=0.5)
        for clf, lbl in zip(classifiers, labels):
            x_test, y_real, y_pred = clf.get_test_values()
            shift_x_test = [i+self.shift*0.1 for i in x_test]
            ax2.plot(self.base_X, self.base_y, label='Curva original')
            ax2.scatter(shift_x_test, y_real, label='Saída esperada')
            ax2.scatter(shift_x_test, y_pred, label='Saída da rede')
            ax3.plot(range(1, len(clf.net.accum_mse) + 1), clf.net.accum_mse, label='MSE')
            ax4.scatter(y_real, y_pred, alpha=0.5)
            ax4.set_xlabel('Saída esperada')
            ax4.set_ylabel('Saída da rede')
            ax4.plot([0,1], [0,1], linestyle='--', color='black', label='curva unitária')
        ax2.legend()
        ax3.legend()
        ax4.legend()
        plt.show()


class NARXNetwork(object):
    def __init__(self, X, y, hidden_layer_sizes=(10,), input_order=0, output_order=0, epochs=300, debug=False):
        self.X = X
        self.y = y
        self.epochs = epochs

        in_nodes = 1
        out_nodes = 1
        incoming_weight_from_output = .4  #Era 0.2
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
        self.net.learn(epochs=self.epochs, show_epoch_results=True, random_testing=False)

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
    pg = PatternGenerator(n_samples=150, debug=debug)

    narx = NARXNetwork(pg.X, pg.y, [5,5], 0, 3, 500)
    narx.fit()
    print(narx.accuracy())

    # pg.better_plot_data([nn for nn in neural_networks], labels)
    pg.better_plot_data([narx], ['NARX 1x5x5x1, 0i3o 1000epochs'])
    # pg.better_plot_data([], [])

if __name__ == '__main__':
    main()


#10,10 e range 100 e 300 epocas

# References
# https://www.mathworks.com/help/deeplearning/ug/design-time-series-narx-feedback-neural-networks.html
# https://en.wikipedia.org/wiki/Nonlinear_autoregressive_exogenous_model