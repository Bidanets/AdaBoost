from sklearn.datasets import make_gaussian_quantiles
import numpy as np
       
class Data_Preprocessor:
    def __init(self):
        return 1
        
    def Prepare_data(self):    
        # Construct dataset
        X1, y1 = make_gaussian_quantiles(cov = 2.0,
                                         n_samples = 200, n_features = 2,
                                         n_classes = 1, random_state = 1)
        X2, y2 = make_gaussian_quantiles(mean = (3, 3), cov=1.5,
                                         n_samples = 300, n_features = 2,
                                         n_classes = 1, random_state = 1)
        
        y2 = [1] * len(y2)
        
        X = np.concatenate((X1, X2))
        y = np.concatenate((y1, y2))
        
        size_of_train_data = int(len(X) * 0.8)
        
        X_train = X[:size_of_train_data]
        y_train = y[:size_of_train_data]
        
        X_test = X[size_of_train_data:]
        y_test = y[size_of_train_data:]
        
        return X_train, y_train, X_test, y_test
    
    
    