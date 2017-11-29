from models.NaiveBayes import NaiveBayes
from models.DecisionTree import DecisionTree
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
import numpy as np
import os

DIR = os.path.dirname(__file__)
DATA_TRAIN = os.path.join(DIR, "../data/speeches.train.liblinear")
DATA_TEST = os.path.join(DIR, "../data/speeches.test.liblinear")

# Read in the data
x_train, y_train = load_svmlight_file(DATA_TRAIN)
x_test, y_test = load_svmlight_file(DATA_TEST)

y_train = np.where(y_train==1, 1., 0.)
y_test = np.where(y_test==1, 1., 0.)

indices = list(range(x_train.shape[1]))
selectedIndices = np.random.choice(indices, 100)
x_train = x_train.tocsc()[:, selectedIndices].toarray()
x_test = x_test.tocsc()[:, selectedIndices].toarray()

print "Train:"
print x_train.shape
print "Test:"
print x_test.shape

featureNames = [str(index) for index in range(x_train.shape[1])]

model = DecisionTree(maxDepth=3)
model.fit(x_train, y_train, featureNames)

predictions = model.predict(x_test, featureNames)

print "Num Positive"
print np.sum(predictions)
print ""
print "Accuracy"
print np.sum(np.where(predictions == y_test, 1. ,0.)) / float(len(y_test))
print precision_recall_fscore_support(y_test, predictions, average="binary")


