import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from neupy.algorithms import GradientDescent, LevenbergMarquardt, ConjugateGradient
from neupy import plots
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class NeuralNetwork(object):
    def __init__(self, X, y, question_a=False):
        if not question_a:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        else:
            X_train = X
            X_test = X
            y_train = y
            y_test = y
        self.X = X
        self.y = y
        self.gradient_trained = False
        self.lm_trained = False
        self.conjugate_trained = False
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def train_gradient(self, epochs, shape):
        self.gradient = GradientDescent(shape, verbose=True)
        self.gradient.train(self.X_train, self.y_train, self.X_test, self.y_test, epochs=epochs)
        self.gradient_trained = True

    def train_levenberg_marquardt(self, epochs, shape):
        self.lmnet = LevenbergMarquardt(shape, verbose=True)
        self.lmnet.train(self.X_train, self.y_train, self.X_test, self.y_test, epochs=epochs)
        self.lm_trained = True

    def train_conjugate(self, epochs, shape):
        self.conjugate = ConjugateGradient(shape, verbose=True)
        self.conjugate.train(self.X_train, self.y_train, self.X_test, self.y_test, epochs=epochs)
        self.conjugate_trained = True

    def plot_gradient_errors(self):
        plots.error_plot(self.gradient)

    def plot_lm_errors(self):
        plots.error_plot(self.lmnet)

    def plot_conjugate_errors(self):
        plots.error_plot(self.conjugate)

    def plot_func_scatter(self):
        fig = plt.figure()
        # ax = Axes3D(fig)
        ax1 = fig.add_subplot(1, 4, 1, projection='3d')
        ax1.set_title('Função')
        ax2 = fig.add_subplot(1, 4, 2, projection='3d')
        ax2.set_title('Gradiente')
        ax3 = fig.add_subplot(1, 4, 3, projection='3d')
        ax3.set_title('Levenberg-Marquadt')
        ax4 = fig.add_subplot(1, 4, 4, projection='3d')
        ax4.set_title('Gradiente conjugado')
        X1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        X2 = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        X3 = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        input_x = np.array([[a, b, c] for (a, b, c) in zip(X1, X2, X3)])
        y = np.array([1, 0, 0, 1, 1, 0, 0, 1]) * 100
        ax1.scatter3D(X1, X2, X3, s=y, depthshade=False)
        if self.gradient_trained:
            y_grad = self.gradient.predict(input_x)*100
            ax2.scatter3D(X1, X2, X3, s=y_grad, depthshade=False)
        if self.lm_trained:
            y_lm = self.lmnet.predict(input_x)*100
            ax3.scatter3D(X1, X2, X3, s=y_lm, depthshade=False)
        if self.conjugate_trained:
            y_conj = self.conjugate.predict(input_x)*100
            ax4.scatter3D(X1, X2, X3, s=y_conj, depthshade=False)
        plt.show()

    def plot_func(self, x_lims, y_func, normalize=False, standardize=False):
        fig = plt.figure()
        ax = Axes3D(fig)
        X1 = np.linspace(x_lims[0], x_lims[1], 100)
        X2 = np.linspace(x_lims[0], x_lims[1], 100)
        X1, X2 = np.meshgrid(X1, X2)
        y_func_vec = np.vectorize(y_func)
        y = y_func_vec(X1, X2)
        if normalize:
            # y = 2*(y - y.min())/y.max() -1
            y = y/y.max()
        if standardize:
            y = (y-y.mean())/y.std()
        ax.plot_wireframe(X1, X2, y, colors=['r'], linewidths=[1], label='Função original')

        X1 = np.reshape(X1, (1, -1))[0]
        X2 = np.reshape(X2, (1, -1))[0]
        X = np.array([(i, j) for (i, j) in zip(X1, X2)])
        if self.gradient_trained:
            y1 = self.gradient.predict(X)
        if self.lm_trained:
            y2 = self.lmnet.predict(X)
        if self.conjugate_trained:
            y3 = self.conjugate.predict(X)
        X1 = np.reshape(X1, (100, 100))
        X2 = np.reshape(X2, (100, 100))
        if self.gradient_trained:
            y1 = np.reshape(y1, (100, 100))
            ax.plot_wireframe(X1, X2, y1, colors=['g'], linewidths=[1], label='Gradiente')
        if self.lm_trained:
            y2 = np.reshape(y2, (100, 100))
            ax.plot_wireframe(X1, X2, y2, linewidths=[1], label='Levenberg-Marquadt')
        if self.conjugate_trained:
            y3 = np.reshape(y3, (100, 100))
            ax.plot_wireframe(X1, X2, y3, colors=['y'], linewidths=[1], label='Gradiente conjugado')
        ax.legend()
        plt.show()

def y_func_b(a, b):
    pi = np.pi
    factor_1 = np.cos(2*pi*a)/(1-(4*a)**2) * np.sin(pi*a)/(pi*a)
    factor_2 = np.cos(2*pi*b)/(1-(4*b)**2) * np.sin(pi*b)/(pi*b)
    return factor_1 * factor_2

def y_func_c(x1, x2):
    pi = np.pi
    m1 = np.array([[0], [0]])
    m2 = np.array([[2], [2]])
    m3 = np.array([[-2], [-2]])
    C = np.array([[1, 0], [0, 1]])
    C_inv = np.linalg.inv(C)
    x = np.array([[x1], [x2]])
    factor_1 = 1/(2*pi)
    parc_1 = -0.5 * ((x-m1).T @ C_inv @ (x-m1))
    parc_2 = -0.5 * ((x-m2).T @ C_inv @ (x-m2))
    parc_3 = -0.5 * ((x-m3).T @ C_inv @ (x-m3))
    factor_2 = np.exp(parc_1) + np.exp(parc_2) + np.exp(parc_3)
    return (factor_1 * factor_2)[0][0]

def y_func_d(x1, x2):
    return (0.5*x1 + 0.5*x2) ** 2

def generate_input_a():
    X = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
    y = np.array([[1], [0], [0], [1], [1], [0], [0], [1]])
    return X, y

def generate_input(x_lims, n_samples, func, normalize=False, standardize=False):
    X1 = np.linspace(x_lims[0], x_lims[1], n_samples)
    X2 = np.linspace(x_lims[0], x_lims[1], n_samples)
    np.random.shuffle(X1)
    np.random.shuffle(X2)
    X = np.array([[i, j] for (i, j) in zip(X1, X2)])
    y = np.array([func(i, j) for (i, j) in X])
    if normalize:
        # y = 2*(y - y.min())/y.max() -1
        y = y/y.max()
    if standardize:
        y = (y-y.mean())/y.std()
    return X, y

def main():
    # function a
    # X, y = generate_input_a()
    # nn_shape = (3, 4, 1)

    # function b
    # x_lims = (-4*np.pi, 4*np.pi)
    # x_lims = (-1, 1)
    # func = y_func_b
    # X, y = generate_input(x_lims, 1000, func)
    # nn_shape = (2, 4, 1)

    # function c
    x_lims = (-5, 5)
    func = y_func_c
    X, y = generate_input(x_lims, 1000, func, normalize=True, standardize=False)
    nn_shape = (2, 4, 1)

    # function d
    # x_lims = (-1, 1)
    # func = y_func_d
    # X, y = generate_input(x_lims, 1000, func)
    # nn_shape = (2, 4, 1)

    nn = NeuralNetwork(X, y, question_a=False)
    nn.train_gradient(100000, nn_shape)  # 20.000 for a, 100.000 for b and c
    nn.plot_gradient_errors()
    # nn.train_levenberg_marquardt(50, nn_shape)  # 50 for a, 200 for b and c
    # nn.plot_lm_errors()
    # nn.train_conjugate(2000, nn_shape)  # 20.000 for a, 100.000 for b and c
    # nn.plot_conjugate_errors()
    # nn.plot_func(x_lims, func, normalize=True, standardize=False)
    
    # test func a
    # y_grad = nn.gradient.predict(X)
    # y_lm = nn.lmnet.predict(X)
    # y_conj = nn.conjugate.predict(X)
    # for i in range(8):
        # print(X[i], y[i], y_grad[i], y_lm[i], y_conj[i])
    # nn.plot_func_scatter()

if __name__ == '__main__':
    main()