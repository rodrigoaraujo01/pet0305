print('Importing Matplotlib')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

print('Importing Numpy')
import numpy as np

print('Importing Scikit Learn')
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm

# print('Importing Keras')
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import RMSprop

print('Importing finished')


class PatternGenerator(object):
    def __init__(self, initial_rate, decline_monthly, debug=False):
        self.debug = debug
        self.initial_rate = initial_rate
        self.decline_monthly = decline_monthly
        X = []
        y = []
        for i in range(6, 366):
            tmp = []
            for j in range(1, 7):
                tmp.append(i - j)
            tmp.sort()
            X.append(tmp)
            y.append(initial_rate * np.exp(decline_monthly/30.4 * i))
        self.X = np.array(X)
        self.y = np.array(y)
        if self.debug: print("X,y generated")

    def generate_X(self, start, end):
        X = []
        for i in range(start, end):
            tmp = []
            for j in range(1, 7):
                tmp.append(i - j)
            tmp.sort()
            X.append(tmp)
        return np.array(X)

    def better_plot_data(self, classifiers, labels):
        X = np.arange(6, 366)
        X_extra = np.arange(366, 732)
        y_extra = self.initial_rate * np.exp(self.decline_monthly/30.4 * X_extra)
        nn_X_extra = self.generate_X(366, 732)

        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(X, self.y)
        ax1.plot(X_extra, y_extra, linestyle='--')
        for clf, lbl in zip(classifiers, labels):
            ax1.plot(X, clf.predict(self.X), label=lbl)
            ax1.plot(X_extra, clf.predict(nn_X_extra), label=lbl+' (previsão)', linestyle='--')
        ax2 = plt.subplot(1, 2, 2)
        for clf, lbl in zip(classifiers, labels):
            ax2.scatter(self.y, clf.predict(self.X), label=lbl+' (entrada)')
            ax2.scatter(y_extra, clf.predict(nn_X_extra), label=lbl+' (pontos extras)')
            points = [min(np.hstack((self.y, y_extra)))-0.1, max(np.hstack((self.y, y_extra)))+0.1]
            ax2.plot(points, points, linestyle='--', color='red', label='m=1')
            # ax1.plot(X, clf.predict(self.X), label=lbl)
            # ax1.plot(X_extra, clf.predict(nn_X_extra), label=lbl+' (previsão)', linestyle='--')
        ax1.legend()
        ax2.legend()
        plt.show()


class NeuralNetwork(object):
    def __init__(self, X, y, hidden_layer_sizes=(10,), debug=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=27)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.clf = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, 
            alpha=1e-3,
            activation='logistic',
            solver='lbfgs',
            verbose=debug,
            random_state=3, # 1 da r2 100% :O
            )

    def fit(self):    
        self.clf.fit(self.X_train, self.y_train)

    def accuracy(self):
        y_pred = self.clf.predict(self.X_test)
        return [mean_squared_error(self.y_test, y_pred), r2_score(self.y_test, y_pred)]

    def predict(self, X):
        return self.clf.predict(X)



def main():
    print("Começou")
    debug = True
    pg = PatternGenerator(
        initial_rate=1,
        decline_monthly=-0.1,
        debug=debug
        )

    # architectures = [(100,100), (300,300,), (400,), (500,), (1000,), (100,100,100), (200,200,200,)]
    architectures = [(50,)]
    labels = [f'6x{"x".join(str(i) for i in arch)}x1' for arch in architectures]
    neural_networks = []

    for arch, lbl in zip(architectures,labels):
        nn = NeuralNetwork(pg.X, pg.y, arch)
        nn.fit()
        mse, r2s = nn.accuracy()
        print(f'{lbl}: mse:{mse} r2_score:{100*r2s:.2f}')
        neural_networks.append(nn)

    pg.better_plot_data([nn.clf for nn in neural_networks], labels)

if __name__ == '__main__':
    main()


# References
# https://stackoverflow.com/questions/42713276/valueerror-unknown-label-type-while-implementing-mlpclassifier
# http://www.machinelearningtutorial.net/2017/01/28/python-scikit-simple-function-approximation/
# https://stackoverflow.com/questions/41308662/how-to-tune-a-mlpregressor
# 