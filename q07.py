print('Importing Matplotlib')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

print('Importing Numpy')
import numpy as np

print('Importing Scikit Learn')
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import svm

print('Importing Keras')
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

print('Importing finished')


class PatternGenerator(object):
    def __init__(self, n_samples=100, debug=False):
        self.debug = debug
        self.n_samples = n_samples
        theta = np.linspace(0, 20, self.n_samples)
        X1a = theta/4 * np.cos(theta)
        X1b = (theta/4 + 0.8) * np.cos(theta)
        self.X1 = np.hstack((X1a, X1b))
        if self.debug: print("X1 generated")

        X2a = theta/4 * np.sin(theta)
        X2b = (theta/4 + 0.8) * np.sin(theta)
        self.X2 = np.hstack((X2a, X2b))
        if self.debug: print("X2 generated")

        self.y = [0 for i in range(self.n_samples)] + [1 for i in range(self.n_samples)]
        if self.debug: print("y generated")

    def x_values(self):
        return np.array([(x1, x2) for (x1, x2) in zip(self.X1, self.X2)])

    def better_plot_data(self, classifiers, labels):
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        cm = plt.cm.RdBu
        X1_space = np.linspace(min(self.X1) - 1, max(self.X1) + 1, self.n_samples)
        X2_space = np.linspace(min(self.X2) - 1, max(self.X2) + 1, self.n_samples)
        xx, yy = np.meshgrid(X1_space, X2_space)
        ax1 = plt.subplot(2, len(classifiers) + 1, 1)
        ax1.scatter(self.X1, self.X2, c=self.y, cmap=cm_bright, s=2)
        # ax2 = plt.subplot(2, len(classifiers) + 1, len(classifiers) + 2)
        # ax2.scatter(self.X1, self.X2, c=self.y, cmap=cm_bright, s=1)
        for i, (clf, lbl) in enumerate(zip(classifiers, labels)):
            ax1 = plt.subplot(2, len(classifiers) + 1, i+2)
            if hasattr(clf, "decision_function"):
                # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            elif hasattr(clf, 'predict'):
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)
            ax1.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            ax1.set_title(lbl)
            # Plot also the points
            # ax1.scatter(self.X1, self.X2, c=self.y, cmap=cm_bright, s=1)

            ax2 = plt.subplot(2, len(classifiers) + 1, i + len(classifiers) + 3)
            X = np.array([(x1,x2) for (x1, x2) in zip(self.X1, self.X2)])
            if 'DLN' not in lbl:
                ax2.scatter(self.X1, self.X2, c=clf.predict(X), cmap=cm_bright, s=1)
            else:
                test = clf.predict(X)
                ax2.scatter(self.X1, self.X2, c=test.T[0], cmap=cm_bright, s=1)
        plt.show()
    
    def square_plot(self, classifiers, labels):
        cm = plt.cm.RdBu
        X1_space = np.linspace(-5, 5, 100)
        X2_space = np.linspace(-5, 5, 100)
        xx, yy = np.meshgrid(X1_space, X2_space)
        for i, (clf, lbl) in enumerate(zip(classifiers, labels)):
            ax1 = plt.subplot(1, len(classifiers), i+1)
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            elif hasattr(clf, 'predict'):
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)
            ax1.imshow(Z, cmap=cm, alpha=.8)
            ax1.set_title(lbl)
        plt.show()

class NeuralNetwork(object):
    def __init__(self, X, y, hidden_layer_sizes=(10,), debug=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, random_state=27)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, 
            max_iter=20000, 
            # alpha=1e-7,  #1e-3: 0.6
            solver='lbfgs',
            verbose=debug,
            random_state=27,
            # tol=0.000000001
            )

    def fit(self):    
        self.clf.fit(self.X_train, self.y_train)

    def accuracy(self):
        y_pred = self.clf.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred), confusion_matrix(self.y_test, y_pred)

    def predict(self, X):
        return self.clf.predict(X)


class SupportVectorMachine(object):
    def __init__(self, X, y, debug=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, random_state=27)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.clf = svm.SVC(kernel='rbf', gamma=2, C=1)

    def fit(self):    
        self.clf.fit(self.X_train, self.y_train)

    def accuracy(self):
        y_pred = self.clf.predict(self.X_test)
        print(y_pred)
        return accuracy_score(self.y_test, y_pred), confusion_matrix(self.y_test, y_pred)
        # return accuracy_score(self.y_test, y_pred)

    def predict(self, X):
        return self.clf.predict(X)


class DeepLearningNetwork(object):
    def __init__(self, X, y, debug=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, random_state=27)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.debug = debug

        self.model = Sequential()
        self.model.add(Dense(16, activation='relu', input_shape=(2,)))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(
            loss='binary_crossentropy',
            optimizer=RMSprop(),
            metrics=['accuracy']
            )

    def fit(self):
        self.history = self.model.fit(
            self.X_train, 
            self.y_train, 
            batch_size=32, 
            epochs=4000, 
            verbose=0)

    def accuracy(self):
        y_pred = self.model.predict(self.X_test)
        y_pred = (y_pred > 0.5)
        return self.model.evaluate(self.X_test, self.y_test, verbose=0), confusion_matrix(self.y_test, y_pred.T[0])


def main():
    print("Começou")
    debug = True
    pg = PatternGenerator(n_samples=100, debug=debug)

    nn0 = NeuralNetwork(pg.x_values(), pg.y, (200,))
    nn0.fit()
    if debug:
        print('NN0 trained')
        print(nn0.accuracy())

    svm = SupportVectorMachine(pg.x_values(), pg.y)
    svm.fit()
    if debug: 
        print('SVM trained')
        print(svm.accuracy())

    dln = DeepLearningNetwork(pg.x_values(), pg.y)
    dln.fit()
    if debug: 
        print('DLN trained')
        print(dln.accuracy())

    pg.square_plot([nn0.clf, svm.clf, dln.model],['NN', 'SVM', 'DLN'])
    pg.better_plot_data([nn0.clf, svm.clf, dln.model],['NN', 'SVM', 'DLN'])
    # pg.better_plot_data([nn0.clf, svm.clf],['2x2000x1', 'SVM',])

if __name__ == '__main__':
    main()


# References
# https://datascience.stackexchange.com/questions/22830/deep-neural-network-using-keras-tensorflow-solves-spiral-dataset-classification
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# https://en.wikipedia.org/wiki/Confusion_matrix
