from __future__ import division
from collections import Counter
import codecs
import math
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from FeatureSelector import *
from sklearn.metrics import precision_score, make_scorer
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import isspmatrix


out_file = codecs.open('out.txt','w',encoding='utf-8')
sense_label = {'NEG': 0, 'POS':1}
class Review:
	def __init__(self, sense):
		self.sense = sense
		self.unigrams = Counter()
		self.bigrams = Counter()

	def set_language_model(self, sent):
		self.unigrams.update(sent)

		bi_list = []
		for i in range(len(sent) - 1):
			bi_list.append(tuple((sent[i], sent[i+1])))

		self.bigrams.update(bi_list)


class ReviewReader():
	def __init__(self):
		self.reviews = []

		# self.unigrams_vocab = []
		self.unigrams_vocab = Counter()
		self.bigrams_vocab = Counter()

		self.categories = Counter()
		self.wordDict = {}
		self.stopwords_list  = ['and', 'the', 'The', 'I', 'we', 'is', 'are', 'am', 'was', 'were', 'my', 'My', 'our', 'Our', 'a', 'A', 'an', 'An', 'in', 'it', 'of', 'for']
		# self.sense_unigrams = {}
		# self.sense_unigrams['NEG'] = Counter()
		# self.sense_unigrams['POS'] = Counter()

		# self.sense_bigrams = {}
		# self.sense_bigrams['NEG'] = Counter()
		# self.sense_bigrams['POS'] = Counter()

	# def update_vocab_language_model(self, contents):
	# 	for w in contents:
	# 		if w not in self.unigrams_vocab:
	# 			self.unigrams_vocab.append(w)


	# def update_sense_language_model(self, sense, contents):
	# 	self.sense_unigrams[sense].update(contents)

	# 	for i in range(len(contents) - 1):
	# 		bigram = tuple((contents[i], contents[i+1]))
	# 		if bigram in self.sense_bigrams[sense]:
	# 			self.sense_bigrams[sense][bigram] += 1
	# 		else:
	# 			self.sense_bigrams[sense][bigram] = 1

	# def get_bigram_list(self, sent):
	# 	bi_list = []
	# 	for i in range(len(sent) - 1):
	# 		bi_list.append(tuple((sent[i], sent[i+1])))
	# 	return bi_list

	def remove_stopwords(self):
		for word in self.stopwords_list:
			if word in list(self.unigrams_vocab):
				del self.unigrams_vocab[word]
			if word in self.wordDict:
				del self.wordDict[word]

	def update_wordDict(self, words, reviewIndex):
		for w in words:
			if w not in self.wordDict:
				self.wordDict[w] = []
			if reviewIndex not in self.wordDict[w]:
				self.wordDict[w].append(reviewIndex)
		return

	def word_cat_num(self, w, ci):
		# wordCatOutfile.write(w + ':\n')
		# wordCatOutfile.write(ci + '||\n')

		sum =0
		if w in self.wordDict: #assertion if#
			for rev_id in self.wordDict[w]:
				if self.reviews[rev_id - 1].sense == ci:
					sum += 1
					# wordCatOutfile.write(str(rev_id) + '\n')
			# wordCatOutfile.write('--' + str(sum) + '\n\n')
		return sum


	def read(self):
		with codecs.open('./dataset/dataset.txt', 'r', encoding = 'utf-8') as f:
			l = 0
			review_count = 0
			for line in f:
				l += 1
				line_contents = line.strip().split()

				sense = line_contents[3]
				if sense == 'NEG' or sense == 'POS':
					# self.categories.update(sense)
					if sense in self.categories:
						self.categories[sense] += 1
					else:
						self.categories[sense] = 1			
					review = Review(sense)
					del line_contents[0:4]
					review.set_language_model(line_contents)

					self.reviews.append(review)
					review_count += 1


					# self.update_vocab_language_model(line_contents)
					self.unigrams_vocab.update(review.unigrams)
					self.bigrams_vocab.update(review.bigrams)
					# self.update_sense_language_model(sense, line_contents)
					self.update_wordDict(line_contents, len(self.reviews))


				if l == 10 or l == 12:
					for i in range(len(line_contents)):
						out_file.write(line_contents[i])
						if i < len(line_contents) - 1:
							out_file.write(' ')
					out_file.write('\n')

		print l
		print review_count


# class FeatureSelector(object):
# 	def __init__(self, rr):
# 		self.rr = rr
# 		self.features = []

# 	def select_features(self):
# 		return self.select()



# class UnigramFeatureSelector(FeatureSelector):
# 	def __init__(self, rr, n):
# 		super(UnigramFeatureSelector, self).__init__(rr)
# 		self.n = n

# 	def select(self):
# 		for word, count in self.rr.unigrams_vocab.most_common(self.n):
# 			self.features.append(word)

# 		return self.features

# class UniBiFeatureSelector(FeatureSelector):
# 	def __init__(self, rr, n, m):
# 		super(UniBiFeatureSelector, self).__init__(rr)
# 		self.n = n
# 		self.m = m

# 	def select(self):
# 		for w, count in self.rr.unigrams_vocab.most_common(self.n):
# 			self.features.append(w)

# 		for t, count in self.rr.bigrams_vocab.most_common(self.m):
# 			self.features.append(t)

# 		return self.features



class LOO:
	def __init__(self, model):
		self.loo = LeaveOneOut()
		self.model = model
		self.pos_tp = 0
		self.pos_fp = 0
		self.pos_fn = 0
		self.neg_tp = 0
		self.neg_fp = 0
		self.neg_fn = 0


	def update_evals(self, y_test, predicted):
		if y_test == [1]:
			if predicted == [1]:
				self.pos_tp += 1
			else:
				self.pos_fn += 1
		else:
			if predicted == [1]:
				self.pos_fp += 1			

		if y_test == [0]:
			if predicted == [0]:
				self.neg_tp += 1
			else:
				self.neg_fn += 1
		else:
			if predicted == [0]:
				self.neg_fp += 1			

	def get_evals(self, acc):
		pos_precision = self.pos_tp / (self.pos_tp + self.pos_fp)
		pos_recall = self.pos_tp / (self.pos_tp + self.pos_fn)

		neg_precision = self.neg_tp / (self.neg_tp + self.neg_fp)
		neg_recall = self.neg_tp / (self.neg_tp + self.neg_fn)

		if abs(acc - 0.8709) < 1e-5:
			print '*************************************************************************'
		print 'accuracy', acc

		print 'positive precision', pos_precision
		print 'positive recall', pos_recall
		print 'positive f1', 2*((pos_precision * pos_recall) / (pos_precision + pos_recall))

		print '\n'

		print 'negative precision', neg_precision
		print 'negative recall', neg_recall
		print 'negative f1', 2*((neg_precision * neg_recall) / (neg_precision + neg_recall))

	def leave_one_out(self, samples, labels):
		acc_sum = 0
		for train_index, test_index in self.loo.split(samples):
			x_train, x_test = samples[train_index[0] : train_index[-1] + 1], samples[test_index[0] : test_index[-1] + 1]
			y_train, y_test = labels[train_index[0] : train_index[-1] + 1], labels[test_index[0] : test_index[-1] + 1]

			self.model.fit(x_train, y_train)

			predicted = model.predict(x_test)

			self.update_evals(y_test, predicted)

			acc_sum += model.score(x_test, y_test)

		acc = acc_sum / len(samples)
		print acc
		print self.pos_tp, self.pos_fp + self.pos_tp, self.pos_fn + self.pos_tp
		print self.neg_tp, self.neg_fp + self.neg_tp, self.neg_fn + self.neg_tp

		print '________________________________'

		self.get_evals(acc)

class GridSearchLOO(LOO):
	def leave_one_out(self, samples, labels):
		self.model.fit(samples, labels)
		print 'scorer', self.model.scorer_
		print 'score', self.model.best_score_
		print 'splits', self.model.n_splits_
		print 'best params', self.model.best_params_


class DataBuilder(object):
	def __init__(self, features):
		self.features = features

	def get_review_items(self, review):
		return review.unigrams + review.bigrams

	def build(self, reviews):
		samples = []
		labels = []
		for review in reviews:
			sample = [0] * len(self.features)
			items = self.get_review_items(review)
			for w in items:
				if w in self.features:
					sample[self.features.index(w)] = self.get_feature_value(w, items, reviews)

			samples.append(sample)
			labels.append(sense_label[review.sense])

		return samples, labels

class OccuranceDataBuilder(DataBuilder):
		def get_feature_value(self, w, review_items, reviews):
			return 1

class FrequencyDataBuilder(DataBuilder):
	def get_feature_value(self, w, review_items, reviews):
		return review_items[w]


class TFIDFDataBuilder(DataBuilder):
	def get_feature_value(self, w, review_items, reviews):
		rf = 0
		for rev in reviews:
			if w in self.get_review_items(rev):
				rf += 1
		return review_items[w] * math.log10(len(reviews)/rf)


if __name__ == "__main__":

	rr = ReviewReader()
	rr.read()
	# rr.remove_stopwords()
	print '>>>>>>>>>>>>>>>>>>>'
	print len(rr.unigrams_vocab.values())
	print '>>>>>>>>>>>>>>>>>>>'
	# fs = MixedMostCommonFeatureSelector(rr, 24)
	# fs = MostCommonFeatureSelector(rr, 50)
	# fs = MostCommonFeatureSelector(rr, len(rr.unigrams_vocab.values()))
	fs = UniBiMostCommonFeatureSelector(rr, 40, 10)

	# fs = UnigramFeatureSelector(rr, 35)
	# fs = UniBiFeatureSelector(rr, 25, 25)

	# sagg = MaxScoreAggregation()
	# fs = XSquareFeatureSelector(rr, 50, sagg)
	# fs = MutualInfo1FeatureSelector(rr, 50, sagg)
	# fs = InfoGainFeatureSelector(rr, 50)


	uni_features = fs.select_features()



	feat_size = len(uni_features)
	print 'fffffffffffffffffff'
	print feat_size


	# data_builder = OccuranceDataBuilder(uni_features)
	data_builder = FrequencyDataBuilder(uni_features)
	# data_builder = TFIDFDataBuilder(uni_features)
	samples, labels = data_builder.build(rr.reviews)
	# if isspmatrix(samples):
	# 	print 'yes'
	# else:
	# 	print 'no'
	# print len(samples)
	# print len(samples[0])
	# samples, s, vh = svds(samples, k = 100)
	# print '###'
	# print 'chitoooooo'
	# print len(samples)
	# print len(samples[0])
	
	samples, labels = shuffle(samples, labels)

	# model = svm.SVC(kernel = 'linear', C = 1)
	params = {'C': [1], 'class_weight': ['balanced'], 'loss': ['squared_hinge']}
	# params = {'C': [1, 10], 'kernel': ['linear']}
	scoring = {'acc': 'accuracy'}
	# model = GridSearchCV(svm.LinearSVC(), params, cv = LeaveOneOut(), refit = 'acc', scoring = scoring)
	# model = svm.LinearSVC(loss = 'squared_hinge', C = 1, class_weight = 'balanced')
	model = MultinomialNB(alpha = 1)
	loo = LOO(model)
	# loo = GridSearchLOO(model)
	loo.leave_one_out(samples, labels)
