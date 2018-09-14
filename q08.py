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

        ax1 = plt.subplot(1, 3, 1)
        ax1.set_title(f'Curva de produção\nQ0={self.initial_rate} m3/d, declínio={self.decline_monthly} a.m.')
        ax2 = plt.subplot(1, 3, 2)
        ax2.set_title('Ampliação dos valores estimados')
        ax1.plot(X, self.y, label='curva real')
        ax1.plot(X_extra, y_extra, linestyle='--', label='curva real (previsão)')
        ax2.plot(X_extra, y_extra, linestyle='--', label='curva real (previsão)')
        for clf, lbl in zip(classifiers, labels):
            nn_y = clf.predict(self.X)
            nn_y_extra = clf.predict(nn_X_extra)
            ax1.plot(X, nn_y, label=lbl)
            ax1.plot(X_extra, nn_y_extra, label=lbl+' (previsão)', linestyle='--')
            ax2.plot(X_extra, nn_y_extra, label=lbl+' (previsão)', linestyle='--')
            error_1 = nn_y - self.y
            error_2 = nn_y_extra - y_extra
            mse_1 = mean_squared_error(nn_y, self.y)
            mse_2 = mean_squared_error(nn_y_extra, y_extra)
            r2s_1 = r2_score(nn_y, self.y)
            r2s_2 = r2_score(nn_y_extra, y_extra)
            print('Erros do conjunto de entrada\n',
                  f'- mse: {mse_1:.4}\n',
                  f'- r2:{r2s_1:.4}\n',
                  f'- média do erro: {error_1.mean():.4}\n',
                  f'- variância do erro: {error_1.var():.4}')
            print('Erros do conjunto de estimativas\n',
                  f'- mse: {mse_2:.4}\n',
                  f'- r2:{r2s_2:.4}\n',
                  f'- média do erro: {error_2.mean():.4}\n',
                  f'- variância do erro: {error_2.var():.4}')
        ax3 = plt.subplot(1, 3, 3)
        ax3.set_title('Erro de predição')
        for clf, lbl in zip(classifiers, labels):
            ax3.scatter(self.y, clf.predict(self.X), label=lbl+' (entrada)', alpha=0.3)
            ax3.scatter(y_extra, clf.predict(nn_X_extra), label=lbl+' (pontos extras)', alpha=0.3)
            points = [min(np.hstack((self.y, y_extra)))-0.1, max(np.hstack((self.y, y_extra)))+0.1]
            ax3.plot(points, points, linestyle='--', color='red', label='m=1')
            # ax1.plot(X, clf.predict(self.X), label=lbl)
            # ax1.plot(X_extra, clf.predict(nn_X_extra), label=lbl+' (previsão)', linestyle='--')
        ax1.legend()
        ax2.legend()
        ax3.legend()
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
        decline_monthly=-0.3,
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
# Prediction error http://www.scikit-yb.org/en/latest/api/regressor/peplot.html
# https://stackoverflow.com/questions/41069905/trouble-fitting-simple-data-with-mlpregressor