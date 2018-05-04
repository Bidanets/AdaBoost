import numpy as np
from sklearn import svm
from sklearn.linear_model import LinearRegression
import optunity.metrics as metrics
from sklearn.neural_network import MLPRegressor as mlpr
from pylab import rcParams
import pandas as pd
import random
import copy
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import matplotlib
import matplotlib.animation as animation
import matplotlib.patches as patches
from pylab import rcParams
from sklearn.linear_model import LogisticRegression
import data_preprocessor
from sklearn.linear_model import Perceptron
import math
from matplotlib.colors import ListedColormap
 
        
class AdaBoost:
    def __init__(self):
        return
            
    # -------------------------------------------------------------------------
    
    
    def draw_Q(self, Q):
        plt.plot(np.arange(1, len(Q)+1), Q, 'ro-', color = 'green', label = 'min Q')

        plt.xlabel('Number of features')
        plt.ylabel('Q')
        plt.title('Q best on each iteration')   
        
        plt.rcParams["figure.figsize"] = (12, 4)
        
        min_Q = min(Q)
        j_optimal = Q.index(min_Q)
        
        plt.plot(j_optimal+1, min_Q, 'ro', color = 'red', label = 'best Q', ms = 8.0)
        
        plt.legend()
        
        plt.show()
    
    # ------------------------------------------------------------------------- 
    
    def fit(self, X_train, y_train, T, noises_percent = 0.0):
        self.T = T
        
        l = len(y_train)
        
        self.w = []
        for i in range(l):
            self.w.append(1/l)
            
        ensemble = []
        alpha = []
            
        self.w = np.array(self.w)
        
        for t in range(T): 
            model = Perceptron(penalty = 'l2', alpha = 0.01, shuffle = True, max_iter = 500)
            b_t = model.fit(X = X_train, y = y_train, sample_weight = self.w)
            
            N = 0.0
            for i in range(l):
                y_hat = model.predict([X_train[i]])
                if (y_hat != y_train[i]):
                    flag = 1
                else:
                    flag = 0
                N += self.w[i] * flag
            alpha_t = 1/2 * math.log((1 + N + 1/l)/(N + 1/l))
            
            ensemble.append(b_t)
            alpha.append(alpha_t)
            
            for i in range(l):
                y = copy.deepcopy(y_train[i])
                
                y_hat = model.predict([X_train[i]])
                
                self.w[i] = self.w[i] * np.exp(-alpha_t * y * y_hat)
                
                w0 = np.sum(self.w)
                for i in range(l):
                    self.w[i] /= w0
                    
            
            W = copy.deepcopy(self.w)
            W = np.sort(W)
            
            
            
            print(W)
            
                    
        self.alpha = copy.deepcopy(alpha)
        self.b = copy.deepcopy(ensemble)
            
        return alpha, ensemble
    
    def predict(self, x):
        ans = []
        for i in range(len(x)):
            R = 0.0
            for t in range(len(self.alpha)):
                R += self.alpha[t] * self.b[t].predict([x[i]])
            ans.append(np.sign(R))
        ans = np.array(ans)        
        return ans
        
        
    # -------------------------------------------------------------------------
    
def draw_classification_map(X_train, y_train, X_test, y_test, adaboost_model):
    for i in range(len(X_train)):
        if y_train[i] == -1:
            col = 'blue'
        else:
            col = 'red'
            
        plt.plot([X_train[i][0]], [X_train[i][1]], 'ro', color = col, markersize = 3.0)
        
       
    cmap_light = ListedColormap(['blue', 'red'])    
    
    h = 0.1
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = adaboost_model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap = cmap_light, alpha = 0.3)

    for i in range(len(y_train)):
        if y_train[i] == -1:
            y_train[i] = 0
    
        
    plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train, cmap = cmap_light,
                edgecolor = 'b', s = 20)
    
    plt.axis('tight')
        
    plt.rcParams["figure.figsize"] = (8, 7)                        
    plt.show()
    

        
def main():
    dp = data_preprocessor.Data_Preprocessor()
    X_train, y_train, X_test, y_test = dp.Prepare_data()
    
    # ----------------------------------
    for i in range(len(y_train)):
        if y_train[i] == 0:
            y_train[i] = -1
            
    for i in range(len(y_test)):
        if y_test[i] == 0:
            y_test[i] = -1
    
    # ----------------------------------
    
    model = AdaBoost()
    model.fit(X_train, y_train, T = 40, noises_percent = 10.0)   
    
    draw_classification_map(X_train, y_train, X_test, y_test, model)
    
    
    
main() 


# try threshold classificator