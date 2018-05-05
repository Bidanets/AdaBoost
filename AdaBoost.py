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
        self.INF = 2E9
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
    
    def calc_Q(self, X, y):
        y_hat = self.predict(X)
        Q = 0.0
        for i in range(len(y)):
            if y[i] != y_hat[i]:
                Q += 1
                
        return Q
        
        
    
    def fit(self, X_train, y_train, T):
        self.T = T
        
        l = len(y_train)
        
        self.w = []
        for i in range(l):
            self.w.append(1.0/l)
            
        ensemble = []
        alpha = []
            
        self.w = np.array(self.w)
        
        Q_prev = self.INF
        stability_iterations = 0
        
        for t in range(T): 
            print('Iteration:', t)
            
            model = Perceptron(max_iter = 1000)
            b_t = model.fit(X = X_train, y = y_train, sample_weight = self.w)
            
            N = 0.0
            for i in range(l):
                y_hat = model.predict([X_train[i]])[0]
                if (y_hat != y_train[i]):
                    flag = 1
                else:
                    flag = 0
                N += self.w[i] * flag
            alpha_t = 1/2 * math.log((1 - N)/N)
            
            ensemble.append(b_t)
            alpha.append(alpha_t)
            
            for i in range(l):
                y = copy.deepcopy(y_train[i])
                
                y_hat = model.predict([X_train[i]])[0]
                
                if y == 0:
                    y = -1
                    
                if y_hat == 0:
                    y_hat= -1
                
                self.w[i] = self.w[i] * np.exp(-alpha_t * y * y_hat)
                
                w0 = np.sum(self.w)
                for i in range(l):
                    self.w[i] /= w0
                    
            
            self.alpha = copy.deepcopy(alpha)
            self.b = copy.deepcopy(ensemble)
            
            self.draw_classification_map(X_train, y_train)
    
            Q = self.calc_Q(X_train, y_train)
            if Q >= Q_prev:
                stability_iterations += 1
                
            Q_prev = copy.deepcopy(Q)
                
            if stability_iterations == 20:
                break
            
            
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
        
        
    def draw_model(self):
        print(self.alpha)
        
        print(len(self.b))
        
        for i in range(len(self.b)):
            a1 = self.b[i].coef_[0][0]
            a2 = self.b[i].coef_[0][1]
            a3 = self.b[i].intercept_[0]
            
            print(a1, a2, a3)
            
            x1 = self.x_min
            y1 = (a1 * self.x_min + a3) / (-a2)
            
            x2 = self.x_max
            y2 = (a1 * self.x_max + a3) / (-a2)
            
            plt.plot([x1, x2], [y1, y2], 'k-')
        
    # -------------------------------------------------------------------------
    
    def draw_classification_map(self, X_train, y_train):
        cmap_light = ListedColormap(['blue', 'red'])    
        
        h = 0.2
        self.x_min, self.x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        self.y_min, self.y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(self.x_min, self.x_max, h), np.arange(self.y_min, self.y_max, h))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
    
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
        
        
        self.draw_model()
        
        
        plt.rcParams["figure.figsize"] = (8, 7)                        
        plt.show()
    

        
def main():
    dp = data_preprocessor.Data_Preprocessor()
    X_train, y_train, X_test, y_test = dp.Prepare_data()
    
    model = AdaBoost()
    model.fit(X_train, y_train, T = 40)   
    
main() 


# try threshold classificator