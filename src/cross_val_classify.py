from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
import codecs
import time


samples2OutFile = codecs.open('samples2.txt','w',encoding='cp720')

CMOutfile = codecs.open('cm.txt','w')

class CrossValidation(object):
	def __init__(self, n, samples, labels):
		self.n = n
		self.samples = samples
		self.labels = labels
		self.confMatrices = []
		self.scores = []

	def cross_validate(self, model):
		self.kfold(model)

	def mean_score(self):
		return np.mean(self.scores)

	def print_conf_matrices(self):
		for cm in self.confMatrices:
			CMOutfile.write(np.array2string(cm, separator=', '))
			CMOutfile.write('\n\n')



class SimpleCrossValidation(CrossValidation):

	def kfold(self, model):
		kf = KFold(n_splits = self.n)
		for train_index, test_index in kf.split(self.samples):
			X_train, X_test = self.samples[train_index[0]:train_index[-1] + 1], self.samples[test_index[0]:test_index[-1] + 1]
			y_train, y_test = self.labels[train_index[0]:train_index[-1] + 1], self.labels[test_index[0]:test_index[-1] + 1]

			model.fit(X_train, y_train)
			preds = model.predict(X_test)

			accuracy = model.score(X_test, y_test)
			self.scores.append(accuracy)

			cm = confusion_matrix(y_test, preds)
			self.confMatrices.append(cm)


class StratifiedCrossValidation(CrossValidation):
	def kfold(self, model):
		self.scores = cross_val_score(model, self.samples, self.labels, cv = self.n)


kfold_policy = raw_input("Simple KFold or Stratified KFold?\n 1- Simple\n 2- Stratified\n")


with open("./resources/dataSamples.txt", "rb") as sampleFile:
	samples = pickle.load(sampleFile)

with open("./resources/dataLabels.txt", "rb") as labelFile:
	labels = pickle.load(labelFile)

#Samples pkl checking
# i = 0
# for sample in samples:
# 	for s in sample:
# 		samples2OutFile.write(str(s) + ' ')

# 	samples2OutFile.write('label: ' + str(labels[i]) + '\n')
# 	i += 1

print '**********************'
model = svm.SVC(kernel = 'linear', C = 1)

# scores = cross_val_score(model, samples, labels, cv = 5)
cv = SimpleCrossValidation(5, samples, labels) if kfold_policy is '1' else StratifiedCrossValidation(5, samples, labels)
classify_start = time.time()
cv.cross_validate(model)
elapsed_classify = time.time() - classify_start

cv.print_conf_matrices()

print 'Classification took ' + str(elapsed_classify) + ' seconds.\n'
print 'Scores of 5 folds:\n'
print cv.scores
print cv.mean_score()



print ('asale mani!')