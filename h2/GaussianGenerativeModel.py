from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class GaussianGenerativeModel:
    def __init__(self, isSharedCovariance=False):
        self.isSharedCovariance = isSharedCovariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __coVariance(self):
        nonsharedcovariance = []
        sharedcovariance = np.zeros((self.features, self.features))
        meandata = []
        for k in range(self.nClasses):
            rows = self.X [self.Y == c]
            if(self.isSharedCovariance == True):
                covariance += np.cov(rows.T) *rows.shape[0]
            else:
                nonsharedcovariance.append(np.cov(rows.T))
            meandata.append(np.mean(rows, axis=0))
        if(self.isSharedCovariance == True):
            return np.array(meandata), shared_cov/self.X.shape[0]
        return np.array(meandata), cov

    # TODO: Implement this method!
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.nClasses = 3
        self.features = X.shape[1]
        self.N = X.shape[0]
        self.covariance = self.__meanCovMatrix()
        countarray = np.zeros(self.nClasses)
        for k in self.Y:
            countarray[k] += 1
        self.b = np.log(countarray / (countarray.sum()))
        return super(GaussianGenerativeModel, self).fit(X, Y)

    def __gaussian(self, x):
        probvector = np.zeros(self.nClasses)
        for k in range(self.nClasses):
            class_probs[c] = multivariate_normal.pdf(x, mean=self.class_means[c], cov= self.coVariance if self.isSharedCovariance else self.covariance[k])
        return np.log(probvector)

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        columns = X_to_predict.shape[0]
        gaussians = np.zeros((columns, self.nClasses))
        for k in range(columns):
            gaussians[k] = self.__gaussian(X_to_predict[i])
        return np.argmax(gaussians + self.b, axis=1)
    def visualize(self, output_file, width=3, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .005), np.arange(y_min,
            y_max, .005))

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
        plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()