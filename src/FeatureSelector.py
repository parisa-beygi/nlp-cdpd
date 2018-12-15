from __future__ import division
import codecs
import math
import operator
from collections import Counter
import HamshahriReader



tfile = codecs.open('tfile.txt','w',encoding='cp720')

# ScoresOutfile = codecs.open('scoresSigma.txt','w',encoding='cp720')
# ScoresOutfile = codecs.open('scoresMax2.txt','w',encoding='cp720')
# ScoresOutfile = codecs.open('scores.txt','w',encoding='cp720')


class ScoreAggregation(object):
	def __init__(self):
		self.categories = Counter()
		self.N = 0

	def set_categories(self, categories):
		self.categories = categories
		self.N = sum(Counter(self.categories).values())

	def word_score(self, w, wcl):
		return self.get_word_score(w, wcl)


class MaxScoreAggregation(ScoreAggregation):
	def get_word_score(self, w, wcl):
		l = sorted(wcl, key = operator.itemgetter(1), reverse = True)
		#word - score of the algorithm - main domain - score of the main domain#
		return tuple((w, l[0][1], l[0][0], l[0][1]))


class SigmaScoreAggregation(ScoreAggregation):

	def get_word_score(self, w, wcl):
		l = sorted(wcl, key = operator.itemgetter(1), reverse = True)

		score = 0

		for ci, si in wcl:
			score += self.categories[ci]*si
		score = score/self.N
		return tuple((w, score, l[0][0], l[0][1]))


class FeatureSelector(object):
	def __init__(self, hr):
		self.hr = hr
		self.wordScoreDict = []
		self.FeaturesOutfile = codecs.open('features.txt','w',encoding='cp720')


	def select_features(self):
		self.select()
		return self.sort_features()

	def sort_features(self):
		self.wordScoreDict = sorted(self.wordScoreDict, key = operator.itemgetter(1), reverse = True)
		return self.build_features()

class MostCommonFeatureSelector(FeatureSelector):
	def __init__(self, hr):
		super(MostCommonFeatureSelector, self).__init__(hr)
		self.ScoresOutfile = codecs.open('MostCommon.txt','w',encoding='cp720')

	def select(self):
		self.wordScoreDict = self.hr.generalWordDict

	def sort_features(self):
		f = []
		for word, count in self.wordScoreDict.most_common(1000):
			f.append(word)
			self.ScoresOutfile.write(word + "\t\t" + str(count) + '\n')
			self.FeaturesOutfile.write(word + '\n')			
		return f

class MutualFeatureSelector(FeatureSelector):
	def __init__(self, hr, scoreAgg):
		self.wordCategoryDict = {}
		super(MutualFeatureSelector, self).__init__(hr)
		scoreAgg.set_categories(hr.categories)
		self.scoreAgg = scoreAgg


	def build_features(self):
		l = self.wordScoreDict[0:100]
		f = []
		col_width = 20  # padding

		for w, s, ci, si in l:
			f.append(w)
			self.ScoresOutfile.write(w + "\t\t" + str(s) + "\t\t" + ci + "\t\t" + str(si)+ "\n")
			# self.ScoresOutfile.write("".join(w.ljust(col_width) + str(s).ljust(col_width) + ci.ljust(col_width) + str(si).ljust(col_width)) + '\n')
			# "".join(word.ljust(col_width) for word in row)
			self.FeaturesOutfile.write(w + '\n')

		return f

	# def word_score(self, w):#Max ci
	# 	wcl = self.wordCategoryDict[w]
	# 	l = sorted(wcl, key = operator.itemgetter(1), reverse = True)
	# 	return tuple((w, l[0][0], l[0][1]))

		
	def select(self):

		N = len(self.hr.documents)

		for w in self.hr.wordDict:

			Nw = len(self.hr.wordDict[w])
			Nwb = N - Nw

			if w not in self.wordCategoryDict:
				self.wordCategoryDict[w] = []


			for ci, ci_count in self.hr.categories.most_common():
				Ni = ci_count
				Nib = N - Ni

				##Todo: check
				Niw = self.hr.word_cat_num(w, ci)

				score = self.calculate(N, Nw, Ni, Niw)

				# # a if condition else b
				# # division by zero is excluded automatically
				# t1 = 0 if Niw == 0 else (Niw/N)*math.log(((N*Niw)/(Nw*Ni)), 2)
				# t2 = 0 if Niwb == 0 else (Niwb/N)*math.log(((N*Niwb)/(Nwb*Ni)), 2)
				# t3 = 0 if Nibw == 0 else (Nibw/N)*math.log(((N*Nibw)/(Nw*Nib)), 2)
				# t4 = 0 if Nibwb == 0 else (Nibwb/N)*math.log(((N*Nibwb)/(Nwb*Nib)), 2)

				# score = t1 + t2 + t3 + t4


				self.wordCategoryDict[w].append(tuple((ci, score)))		

			self.wordScoreDict.append(self.scoreAgg.word_score(w, self.wordCategoryDict[w]))



class MutualInfo2FeatureSelector(MutualFeatureSelector):
	def __init__(self, hr, scoreAgg):
		super(MutualInfo2FeatureSelector, self).__init__(hr, scoreAgg)
		self.ScoresOutfile = codecs.open('MutualInfo_2_' + (scoreAgg.__class__.__name__).replace('ScoreAggregation', '') + '.txt','w',encoding='cp720')



	def calculate(self, N, Nw, Ni, Niw):

		Nwb = N - Nw
		Nib = N - Ni

		Niwb = Ni - Niw
		Nibw = Nw - Niw
		Nibwb = Nib - Nibw
		t1 = 0 if Niw == 0 else (Niw/N)*math.log(((N*Niw)/(Nw*Ni)), 2)
		t2 = 0 if Niwb == 0 else (Niwb/N)*math.log(((N*Niwb)/(Nwb*Ni)), 2)
		t3 = 0 if Nibw == 0 else (Nibw/N)*math.log(((N*Nibw)/(Nw*Nib)), 2)
		t4 = 0 if Nibwb == 0 else (Nibwb/N)*math.log(((N*Nibwb)/(Nwb*Nib)), 2)

		return t1 + t2 + t3 + t4


class MutualInfo1FeatureSelector(MutualFeatureSelector):
	def __init__(self, hr, scoreAgg):
		super(MutualInfo1FeatureSelector, self).__init__(hr, scoreAgg)
		self.ScoresOutfile = codecs.open('MutualInfo_1_' + (scoreAgg.__class__.__name__).replace('ScoreAggregation', '') + '.txt','w',encoding='cp720')

	def calculate(self, N, Nw, Ni, Niw):
		t1 = 0 if Niw == 0 else (Niw/N)*math.log(((N*Niw)/(Nw*Ni)), 2)
		return t1

class XSquareFeatureSelector(MutualFeatureSelector):
	def __init__(self, hr, scoreAgg):
		super(XSquareFeatureSelector, self).__init__(hr, scoreAgg)
		self.ScoresOutfile = codecs.open('X-Square_' + (scoreAgg.__class__.__name__).replace('ScoreAggregation', '')+ '.txt','w',encoding='cp720')

	def calculate(self, N, Nw, Ni, Niw):
		Nwb = N - Nw
		Nib = N - Ni

		Niwb = Ni - Niw
		Nibw = Nw - Niw
		Nibwb = Nib - Nibw

		return (N*pow((Niw*Nibwb) - (Niwb*Nibw), 2))/((Niw + Niwb)*(Nibw + Nibwb)*(Niw + Nibw)*(Niwb + Nibwb))



class InfoGainFeatureSelector(FeatureSelector):
	def __init__(self, hr):
		super(InfoGainFeatureSelector, self).__init__(hr)
		self.ScoresOutfile = codecs.open('InfoGain.txt','w',encoding='cp720')

	def build_features(self):
		l = self.wordScoreDict[0:100]		
		f = []
		for w, s in l:
			f.append(w)
			self.ScoresOutfile.write(w + '\t\t' + str(s) + '\n')
			self.FeaturesOutfile.write(w + '\n')
		return f

	def select(self):
		# for w in self.hr.wordDict:
		# 	for ci, ci_count in self.hr.categories.most_common():
		# 		Niw = self.hr.word_cat_num(w, ci)


		N = len(self.hr.documents)

		for w in self.hr.wordDict:
			wordScore = 0
			s1 = 0
			s2 = 0
			s3 = 0

			Nw = len(self.hr.wordDict[w]) #docs containing w
			Nwb = N - Nw
			Pw = Nw/N
			Pwb = Nwb/N

			for ci, ci_count in self.hr.categories.most_common():
				Ni = ci_count
				Niw = self.hr.word_cat_num(w, ci)
				Niwb = Ni - Niw

				Pci = Ni/N
				Pci_w = Niw/Nw
				Pci_wb = 0
				# if Nwb == 0:
				# 	#so Niwb is also zero#
				# 	Pci_wb = 0
				if Nwb != 0:
					Pci_wb = Niwb/Nwb

				s1 += -Pci*math.log(Pci, 2)

				if Pci_w != 0:
					s2 += Pci_w*math.log(Pci_w, 2)

				if Pci_wb != 0:
					s3 += Pci_wb*math.log(Pci_wb, 2)

			#assertion if#
			# if w in wordScoreDict:
			# 	print "duplicate error in wordScoreDict due to duplication in wordDict!\n"

			self.wordScoreDict.append(tuple((w, s1 + Pw*s2 + Pwb*s3)))
			# ScoresOutfile.write(w + ':\n' + str(wordScore) + '\n&&&\n\n')	
		

