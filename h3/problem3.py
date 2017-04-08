
import numpy as np 
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import matplotlib.colors as c
from Perceptron import Perceptron
from sklearn.svm import SVC

# Implement this class
class KernelPerceptron(Perceptron):
	def __init__(self, numsamples):
		self.numsamples = numsamples
		# Initialize Self.S
		self.S = {} 

	# Implement this!
	def fit(self, X, Y):
		self.X = X
		self.Y = Y
		assert(X.shape[0] == Y.shape[0])

		numsamples = X.shape
		self.bias = 0
		chooseRandom = np.random.randint(X.shape[0], size = self.numsamples)

		for k in chooseRandom:
			x_k = self.X[k]
			y_k = self.Y[k]
			sum = 0
			for i,j in self.S.iteritems():
				x_i = self.X[i]
				sum += j * np.dot(x_i, x_k)
			if y_k * sum <= 0:
				self.S[k] = y_k

	# Implement this!
	def returnprediction(self, x):
		sum = 0 
		for i,j in self.S.iteritems():
				x_i = self.X[i]
				sum += j * np.dot(x_i, x)
		if sum > 0:
			return 1
		else:
			return 0

	def predict(self, X):
		return np.array([self.returnprediction(x) for x in X])


# Implement this class

class BudgetKernelPerceptron(Perceptron):
	def __init__(self, beta, N, numsamples):
		self.beta = beta
		self.N = N
		self.numsamples = numsamples
		self.S = {}
	def removal(self):
		arg_max = None
		for k,l in self.S.iteritems():
			x_k = self.X[k]
			sum = 0 
			for i,j in self.S.iteritems():
				x_i = self.X[i]
				sum += j * np.dot(x_i, x_k)
			if j * sum - j * np.dot(x_k,x_k) < - 1e100:
				arg_max = k 
				del Self.s[arg_max]
	# Implement this!
	def fit(self, X, Y):
		self.X = X
		self.Y = Y
		assert(X.shape[0] == Y.shape[0])

		numsamples = X.shape
		chooseRandom = np.random.randint(X.shape[0], size = self.numsamples)

		for k in chooseRandom:
			x_k = self.X[k]
			y_k = self.Y[k]
			sum = 0
			for i,j in self.S.iteritems():
				x_i = self.X[i]
				sum += j * np.dot(x_i, x_k)
			if y_k * sum <= self.beta:
				self.S[k] = y_k
			if self.N <= len(self.S):
				self.removal()

	# Implement this!
	def returnprediction(self, x):
		sum = 0 
		for i,j in self.S.iteritems():
				x_i = self.X[i]
				sum += j * np.dot(x_i, x)
		if sum > 0:
			return 1
		else:
			return 0

	def predict(self, X):
		return np.array([self.returnprediction(x) for x in X])


# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0
N = 50
numsamples = 2000

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'
clf_file_name = 'clf.png'
# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
k = KernelPerceptron(numsamples)
k.fit(X,Y)
k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)

bk = BudgetKernelPerceptron(beta, N, numsamples)
bk.fit(X, Y)
bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)

X_train = X[2000:]
"""
clf = SVC()
clf.fit(X_train, Y) 
clf.visualize(clf_file_name, width=0, show_charts=True, save_fig=True, include_points=False)
"""
print k.score(X,Y)
#print bk.score(X,Y)
print clf.score(X,Y) 