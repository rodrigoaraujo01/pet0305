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

    def better_plot_data(self, classifiers, labels):
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        cm = plt.cm.RdBu
        X = np.arange(6, 366)
        ax1 = plt.subplot(1, 1, 1)
        ax1.plot(X, self.y)
        for i, (clf, lbl) in enumerate(zip(classifiers, labels)):
            if 'DLN' not in lbl:
                ax1.plot(X, clf.predict(self.X), label=lbl)
            # else:
                # test = clf.predict(X)
                # ax1.scatter(self.X1, self.X2, c=test.T[0], cmap=cm_bright, s=1)
        ax1.legend()
        plt.show()


class NeuralNetwork(object):
    def __init__(self, X, y, hidden_layer_sizes=(10,), debug=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, random_state=27)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.clf = MLPRegressor(
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

    nn0 = NeuralNetwork(pg.X, pg.y, (10,))
    nn0.fit()
    if debug:
        print('NN0 trained')
        print(nn0.accuracy())

    nn1 = NeuralNetwork(pg.X, pg.y, (100,))
    nn1.fit()
    if debug:
        print('NN1 trained')
        print(nn1.accuracy())

    nn2 = NeuralNetwork(pg.X, pg.y, (500,))
    nn2.fit()
    if debug:
        print('NN2 trained')
        print(nn2.accuracy())

    nn3 = NeuralNetwork(pg.X, pg.y, (1000,))
    nn3.fit()
    if debug:
        print('NN3 trained')
        print(nn3.accuracy())


    pg.better_plot_data([nn0.clf, nn1.clf, nn2.clf, nn3.clf],['6x10x1', '6x100x1', '6x500x1', '6x1000x1'])

if __name__ == '__main__':
    main()


# References
# https://stackoverflow.com/questions/42713276/valueerror-unknown-label-type-while-implementing-mlpclassifier
# http://www.machinelearningtutorial.net/2017/01/28/python-scikit-simple-function-approximation/
# https://stackoverflow.com/questions/41308662/how-to-tune-a-mlpregressor
# 