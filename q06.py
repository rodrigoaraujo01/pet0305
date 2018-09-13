import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import svm
# from neupy.algorithms import GradientDescent, LevenbergMarquardt
# from neupy import plots

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

    def better_plot_data(self, classifiers):
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        cm = plt.cm.RdBu
        X1 = np.linspace(-1, 1, self.n_samples)
        X2 = np.linspace(-1, 1, self.n_samples)
        xx, yy = np.meshgrid(X1, X2)
        ax1 = plt.subplot(2, len(classifiers) + 1, 1)
        ax1.scatter(self.X1, self.X2, c=self.y, cmap=cm_bright, s=1)
        # ax2 = plt.subplot(2, len(classifiers) + 1, len(classifiers) + 2)
        # ax2.scatter(self.X1, self.X2, c=self.y, cmap=cm_bright, s=1)
        for i, clf in enumerate(classifiers):
            ax1 = plt.subplot(2, len(classifiers) + 1, i+2)
            if hasattr(clf, "decision_function"):
                # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)
            ax1.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            # Plot also the points
            # ax1.scatter(self.X1, self.X2, c=self.y, cmap=cm_bright, s=1)

            ax2 = plt.subplot(2, len(classifiers) + 1, i + len(classifiers) + 3)
            X = [(x1,x2) for (x1, x2) in zip(self.X1, self.X2)]
            ax2.scatter(self.X1, self.X2, c=clf.predict(X), cmap=cm_bright, s=1)
        plt.show()


class NeuralNetwork(object):
    def __init__(self, X, y, debug=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, random_state=27)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.clf = MLPClassifier(
            solver='lbfgs', 
            hidden_layer_sizes=(100,), 
            random_state=1
            )

    def fit(self):    
        self.clf.fit(self.X_train, self.y_train)

    def accuracy(self):
        y_pred = self.clf.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)

    def predict(self, X):
        return self.clf.predict(X)


class SupportVectorMachine(object):
    def __init__(self, X, y, debug=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, random_state=27)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.clf = svm.SVC(
            kernel='rbf',
            gamma=10,
            C=1
            )

    def fit(self):    
        self.clf.fit(self.X_train, self.y_train)

    def predict(self, X):
        return self.clf.predict(X)

    def accuracy(self):
        y_pred = self.clf.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)

    def optimize(self):
        # {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100]}]
        clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5,
                            scoring='recall_macro', verbose=2)
        clf.fit(self.X, self.y)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()



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

    # pg.plot_data([(y1, 'r'), (y2, 'g')])

    nn = NeuralNetwork(pg.x_values(), pg.y)
    nn.fit()
    if debug: 
        print('NN trained')    
        print(nn.accuracy())

    svm = SupportVectorMachine(pg.x_values(), pg.y)
    # svm.optimize()
    svm.fit()
    if debug: 
        print('SVM trained')    
        print(svm.accuracy())

    pg.better_plot_data([nn.clf, svm.clf])

if __name__ == '__main__':
    main()


# Classifier comparison: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# SVC Documentation http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
# 1.17. Neural network models (supervised) http://scikit-learn.org/stable/modules/neural_networks_supervised.html
# MLPClassifier Documentation http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# GridSearch http://scikit-learn.org/stable/modules/grid_search.html#grid-search
# GridSearchCV Documentation http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# Parameter estimation using grid search with cross-validation http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
