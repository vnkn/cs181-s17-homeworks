
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
from scipy.misc import logsumexp

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class LogisticRegression:
    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    
    def __gradientloss(self):
        softmax = self.X.dot(self.W.T) - logsumexp(self.X.dot(self.W.T), axis = 1)[:, np.newaxis]
        print self.W.shape
        print self.X.shape
        print softmax.shape
        print self.C.shape
        print self.C.T.dot(self.X).shape
        gradient = ((np.exp(softmax) - self.C).T).dot(self.X)
        error = 2 * self.lambda_parameter * self.W
        print len(gradient)
        print len(gradient[0])
        print len(error)
        print self.W.shape
        print self.X.shape
        return gradient +  error

    def __logisticiteration(self):
        self.W -= (self.__gradientloss() * self.eta)
        return self.__gradientloss() 
    

    # TODO: Implement this method!
    def fit(self, X, C):
        X = np.append(X,np.ones((X.shape[0], 1)), axis=1)
        self.X = X
        self.C = C
        self.features = X.shape[1]
        self.W = np.zeros((59,self.features))
        ys = []
        for k in C:
            y = np.zeros(3)
            y = 1
            ys.append(y)
        self.y = np.array(ys)
        norm = 0
        for k in range(1000):
            gradient = self.__logisticiteration()
            norm += np.linalg.norm(gradient)
        print norm
    # TODO: Implement this method!
    def predict(self, X_to_predict): 
        maximized = np.argmax(X_to_predict.dot(self.W.T), axis=1)
        return maximized
    def visualize(self, output_file, width=2, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min,
            y_max, .05))

        # Flatten the grid so the values match spec for self.predict
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        X_topredict = np.vstack((xx_flat,yy_flat)).T

        # Get the class predictions
        Y_hat = self.predict(X_topredict)
        Y_hat = Y_hat.reshape((xx.shape[0], xx.shape[1]))
        
        cMap = c.ListedColormap(['r','b','g'])

        # Visualize them.
        plt.figure()
        plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.C, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
