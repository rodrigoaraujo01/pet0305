import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from neupy.algorithms import GradientDescent, LevenbergMarquardt
from neupy import plots

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Circle(object):
    def __init__(self, center:Point, radius):
        self.center = center
        self.radius = radius
    
    def point_inside(self, p:Point):
        return (p.x - self.center.x)**2 + (p.y - self.center.y)**2 <= self.radius**2


class PatternGenerator(object):
    def __init__(self, n_samples=100, debug=False):
        self.debug = debug
        self.n_samples = 100
        rad = 1.1

        self.circles = [Circle(Point(-1, 0), rad), Circle(Point(0, -1), rad), Circle(Point(1, 0), rad), Circle(Point(0, 1), rad)]
        if self.debug: print("Circles generated")
        
        X1 = []
        X2 = []
        for x1 in np.linspace(-1, 1, self.n_samples):
            for x2 in np.linspace(-1, 1, self.n_samples):
                X1.append(x1)
                X2.append(x2)
        self.X1 = np.array(X1)
        self.X2 = np.array(X2)
        np.random.shuffle(self.X1)
        np.random.shuffle(self.X2)
        if self.debug: print("X1 and X2 generated")
        
        self.y = self.generate_classes()
        if self.debug: print("y generated")

    def x_values(self):
        return np.array([(x1, x2) for (x1, x2) in zip(self.X1, self.X2)])

    def generate_classes(self):
        y = []
        for x1, x2 in zip(self.X1, self.X2):
            p = Point(x1, x2)
            inside = [c.point_inside(p) for c in self.circles]
            if sum(inside) > 1:
                y.append(1)
            else:
                y.append(0)
        return np.array(y)

    def plot_data(self, extra_results):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter3D(self.X1, self.X2, self.y, c='b', depthshade=False)
        for y, c in extra_results:
            ax.scatter3D(self.X1, self.X2, y, c=c, depthshade=False)
        plt.show()


# class NeuralNetwork(object):
#     def __init__(self, X, y):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#         self.X = X
#         self.y = y
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test

#     def train_gradient(self, epochs, shape):
#         self.gradient = GradientDescent(shape, verbose=True)
#         self.gradient.train(self.X_train, self.y_train, self.X_test, self.y_test, epochs=epochs)
#         self.gradient_trained = True

#     def plot_gradient_errors(self):
#         plots.error_plot(self.gradient)

#     def gradient_predict(self, X):
#         return self.gradient.predict(X)

#     def train_levenberg_marquardt(self, epochs, shape):
#         self.lmnet = LevenbergMarquardt(shape, verbose=True)
#         self.lmnet.train(self.X_train, self.y_train, self.X_test, self.y_test, epochs=epochs)

#     def plot_lm_errors(self):
#         plots.error_plot(self.lmnet)

#     def lm_predict(self, X):
#         return self.lmnet.predict(X)


class NeuralNetwork(object):
    def __init__(self, X, y, debug=False):
        self.X = X
        self.y = y
        self.mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,), random_state=1)

    def fit(self):    
        self.mlp.fit(self.X, self.y)

    def predict(self, X):
        return self.mlp.predict(X)


class SupportVectorMachine(object):
    def __init__(self, X, y, debug=False):
        self.X = X
        self.y = y
        self.clf = svm.SVC()

    def fit(self):    
        self.clf.fit(self.X, self.y)

    def predict(self, X):
        return self.clf.predict(X)



def main():
    debug = True
    pg = PatternGenerator(n_samples=100, debug=debug)
    
    # nn = NeuralNetwork(pg.x_values(), pg.y)
    # nn.train_gradient(10000, (2,4,4,1))
    # y1 = nn.gradient_predict(pg.x_values())
    # nn.plot_gradient_errors()
    # nn.train_levenberg_marquardt(40, (2,4,1))
    # y1 = nn.lm_predict(pg.x_values())
    # nn.plot_lm_errors()

    nn = NeuralNetwork(pg.x_values(), pg.y)
    nn.fit()
    y1 = nn.predict(pg.x_values())
    if debug: print('NN trained')

    svm = SupportVectorMachine(pg.x_values(), pg.y)
    svm.fit()
    y2 = svm.predict(pg.x_values())
    if debug: print('SVM trained')
    
    pg.plot_data([(y1, 'r'), (y2, 'g')])


if __name__ == '__main__':
    main()