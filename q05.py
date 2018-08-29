import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from neupy.algorithms import GradientDescent, LevenbergMarquardt
from neupy import plots
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class NeuralNetwork(object):
    def __init__(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def train_gradient(self):
        self.gradient = GradientDescent((2, 5, 5, 1), verbose=True, step=0.1)
        self.gradient.train(self.X_train, self.y_train, self.X_test, self.y_test, epochs=200)

    def train_levenberg_marquardt(self):
        self.lmnet = LevenbergMarquardt((2, 3, 1), verbose=True)
        self.lmnet.train(self.X_train, self.y_train, self.X_test, self.y_test)

    def plot_errors(self):
        plots.error_plot(self.gradient)

    def plot_func(self, y_func):
        fig = plt.figure()
        ax = Axes3D(fig)
        X1 = np.linspace(-1, 1, 100)
        X2 = np.linspace(-1, 1, 100)
        X1, X2 = np.meshgrid(X1, X2)
        y_func_vec = np.vectorize(y_func)
        y = y_func_vec(X1, X2)
        print(X1.shape, X2.shape, y.shape)
        # ax.plot_surface(X1, X2, y, cmap=cm.viridis)
        ax.plot_wireframe(X1, X2, y)
        # plt.show()
        X1a = np.reshape(X1, (1, -1))[0]
        X2a = np.reshape(X2, (1, -1))[0]
        Xa = np.array([(i,j) for (i,j) in zip(X1a, X2a)])
        y = self.gradient.predict(Xa)
        X1a = np.reshape(X1a, (100, 100))
        X2a = np.reshape(X2a, (100, 100))
        y = np.reshape(y, (100, 100))
        ax.plot_wireframe(X1a, X2a, y)
        plt.show()

def y_func_b(a, b):
    pi = np.pi
    factor_1 = np.cos(2*pi*a)/(1-(4*a)**2) * np.sin(pi*a)/(pi*a)
    factor_2 = np.cos(2*pi*b)/(1-(4*b)**2) * np.sin(pi*b)/(pi*b)
    return factor_1 * factor_2

def y_func_c(x1, x2):
    pi = np.pi
    m1 = np.array([[0], [0]])
    m2 = np.array([[0.5], [0.5]])
    m3 = np.array([[-0.5], [-0.5]])
    C = np.array([[1, 0], [0, 1]])
    x = np.array([[x1], [x2]])
    factor_1 = 1/(2*pi)
    parc_1 = -0.5 * (np.matrix.transpose(x-m1) @ np.linalg.inv(C) @ (x-m1))
    parc_2 = -0.5 * (np.matrix.transpose(x-m2) @ np.linalg.inv(C) @ (x-m2))
    parc_3 = -0.5 * (np.matrix.transpose(x-m3) @ np.linalg.inv(C) @ (x-m3))
    factor_2 = np.exp(parc_1) + np.exp(parc_2) + np.exp(parc_3)
    return factor_1 * factor_2

def y_func_d(x1, x2):
    return 0.5*x1 + 0.5*x2

def generate_input_a():
    X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], 
                  [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])
    return X, y

def generate_input_b():
    pi = np.pi
    X1 = np.linspace(-4*pi, 4*pi, 100)
    X2 = np.linspace(-4*pi, 4*pi, 100)
    np.random.shuffle(X1)
    np.random.shuffle(X2)
    X = np.array([[i, j] for (i, j) in zip(X1, X2)])
    y = np.array([y_func_b(i, j) for (i, j) in zip(X1, X2)])
    
    return X, y

def generate_input_c():
    X1 = np.linspace(-10, 10, 100)
    X2 = np.linspace(-10, 10, 100)
    np.random.shuffle(X1)
    np.random.shuffle(X2)
    X = np.array([[i, j] for (i, j) in zip(X1, X2)])
    y = np.array([y_func_c(i, j)[0][0] for (i, j) in zip(X1, X2)])
    return X, y

def generate_input_d():
    X1 = np.linspace(-1, 1, 100)
    X2 = np.linspace(-1, 1, 100)
    np.random.shuffle(X1)
    np.random.shuffle(X2)
    X = np.array([[i, j] for (i, j) in zip(X1, X2)])
    y = np.array([y_func_d(i, j) for (i, j) in zip(X1, X2)])
    return X, y


def main():
    # function a
    # X, y = generate_input_a()
    # function b
    # X, y = generate_input_b()
    # function c
    # X, y = generate_input_c()
    # function d
    X, y = generate_input_d()
    nn = NeuralNetwork(X, y)
    nn.train_gradient()
    nn.plot_errors()
    nn.plot_func(y_func_d)
    # nn.train_levenberg_marquardt()

if __name__ == '__main__':
    main()