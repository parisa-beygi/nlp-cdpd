from __future__ import division
from collections import Counter
import codecs

# directory = '/dataset/HAM2-corpus-short-with-tag-selected.txt'
DOC_SIGN = '@@@@@@@@@@'
outfile = codecs.open('out.txt','w',encoding='cp720')
# domainFile = codecs.open('out2.txt','w',encoding='cp720')
wordDictOutfile = codecs.open('out3.txt','w',encoding='cp720')
# wordDictOutfile_docs = codecs.open('out4.txt','w',encoding='cp720')
catOutfile = codecs.open('categories.txt','w',encoding='cp720')
wordCatOutfile = codecs.open('wordcat.txt','w',encoding='cp720')

class Document:
	def __init__(self, domain):
		self.domain = domain
		self.dictionary = Counter()

	def printDoc(self):
		outfile.write(self.domain + ':\n')
		for word, count in self.dictionary.most_common():
			outfile.write(word + ' ' + str(count) + '\n')
		outfile.write(str(self.word_count()))
		outfile.write('\n---------------------------------------------------------------------')	

		return

	def word_count(self):
		sum = 0
		for word, count in self.dictionary.most_common():
			sum += count
		return sum



class HamshahriReader:
	def __init__(self):
		self.wordDict = {}
		self.documents = []
		self.generalWordDict = Counter()
		self.categories = Counter()


	def update_wordDict(self, words, docIndex):
		for w in words:
			if w not in self.wordDict:
				self.wordDict[w] = []
			if docIndex not in self.wordDict[w]:
				self.wordDict[w].append(docIndex)
		return

	def print_list(self, l):
		for d in l:
			wordDictOutfile.write(str(d) + ',')
		wordDictOutfile.write('\n')
		return

	def print_wordDict(self):
		for w in self.wordDict:
			wordDictOutfile.write(w + ':\n')
			wordDictOutfile.write(str(len(self.wordDict[w])) + " -------\n")
			self.print_list(self.wordDict[w])
	  #       for d in wordDict[w]:
			# 	wordDictOutfile.write(str(d) + '* ')
			# # wordDictOutfile.write("\n")
		return

	def print_categories(self):
		for word, count in self.categories.most_common():
			catOutfile.write(word + ': ')
			catOutfile.write(str(count) + '\n')

		return 

	def word_cat_num(self, w, ci):
		wordCatOutfile.write(w + ':\n')
		wordCatOutfile.write(ci + '||\n')

		sum =0
		if w in self.wordDict: #assertion if#
			for doc_id in self.wordDict[w]:
				if self.documents[doc_id - 1].domain == ci:
					sum += 1
					wordCatOutfile.write(str(doc_id) + '\n')
			wordCatOutfile.write('--' + str(sum) + '\n\n')
		return sum


	def extract_domain(self, l):
		l[-1] = l[-1].replace(DOC_SIGN, '')
		return ' '.join(l)

	def read(self, dr):
		with codecs.open(dr,'r',encoding='cp720') as f:
		    for line in f:
		    	# wordDictOutfile_docs.write(line)
		    	# wordDictOutfile_docs.write("____________________________________________________\n")
		    	contents = line.strip().split()
		    	for i in range(len(contents)):
		    		if DOC_SIGN in contents[i]:
		    			# domainFile.write(contents[i] + '\n')
		    			cat = self.extract_domain(contents[0:(i+1)])
		    			del contents[0:(i+1)]
		    			# dic = Counter()
		    			self.documents.append(Document(cat))
		    			if cat in self.categories:
		    				self.categories[cat] += 1
		    			else:
		    				self.categories[cat] = 1
		    			break
		    	self.documents[-1].dictionary.update(contents)
		    	self.update_wordDict(contents, len(self.documents))
		    	self.generalWordDict.update(contents)




