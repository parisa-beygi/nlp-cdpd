from __future__ import division
import codecs
from nltk.tag import hmm
from nltk.metrics import ConfusionMatrix
import numpy as np

out_file = codecs.open('out4.txt','w',encoding='cp720')

cm_out_file = codecs.open('POS_normalized_cm.txt','w',encoding='cp720')
sents_tagged_out_file = codecs.open('sents_tagged_out.txt','w',encoding='cp720')

def construct_label_sequence(file_path):
	document = []
	sent_sizes = []
	sentence = []
	with codecs.open(file_path,'r',encoding='cp720') as f:
	    for line in f:
	    	contents = line.strip().split()
	    	if 'DELM' not in contents:
		    	word = ''.join(contents[0:-1])
		    	sentence.append(tuple((word, contents[-1])))
	    	else:
	    		if sentence:
		    		document.append(sentence)
	    			sent_sizes.append(len(sentence))
	    		sentence = []

	    if 'DELM' not in contents and sentence:
	    	document.append(sentence)
	    	sent_sizes.append(len(sentence))

	# print sent_sizes
	return document


def tag_list(tagged_sents):
	return [tag for sent in tagged_sents for (word, tag) in sent]

def sent_tag_list(tagged_sent):
	return [tag for (word, tag) in tagged_sent]

def sent_word_list(sent):
	return [word for (word, tag) in sent]


def check_tag(doc):
	for i in range(len(doc)):
		for (word, tag) in doc[i]:
			if tag == u'0':
				out_file.write(word + '\n')
				out_file.write(str(i) + '\n')


def get_cm_row_count(cm, row):
	cm_values = cm._values

	row_count = 0
	for j in range(len(cm_values)):
		v1 = cm_values[row]
		v2 = cm_values[j]
		row_count += cm.__getitem__((v1, v2))

	return row_count

def get_normalized_cm(cm):
	cm_values = cm._values
	dim = len(cm_values)
	# normalized_cm = [[0]*dim] * dim
	normalized_cm = []
	for i in range(dim):
		row_count = get_cm_row_count(cm, i)
		l = []
		for j in range(dim):
			l.append(round(cm.__getitem__((cm_values[i], cm_values[j])) / row_count, 2))
		normalized_cm.append(l)
			# print ('for %d th row: ' % i)
			# print cm.__getitem__((cm_values[i], cm_values[j]))
			# print row_count
			# normalized_cm[i][j] = round(cm.__getitem__((cm_values[i], cm_values[j])) / row_count, 4)
			# if i == 1:
			# 	print 'inja'
			# 	print cm.__getitem__((cm_values[i], cm_values[j])) / row_count
			# 	print normalized_cm[i][j]
	# print normalized_cm
	# l = np.array(normalized_cm)
	# cm_out_file.write(np.array2string(l, max_line_width = 200, separator=', '))

	return normalized_cm

def print_cm_accuracy(cm):
	total = 0
	trues = 0

	cm_values = cm._values

	for i in range(len(cm_values)):
		for j in range(len(cm_values)):
			v1 = cm_values[i]
			v2 = cm_values[j]
			count = cm.__getitem__((v1, v2))
			total += count 

			if v1 == v2:
				trues += count
	print trues, total
	print ('accuracy with respect to confusion matrix: %.2f' % (100*trues/total))			


if __name__ == "__main__":

	train_file_path = './dataset/POStrutf.txt'
	train_document = construct_label_sequence(train_file_path)

	check_tag(train_document)

	print 'train tagset_______'
	print set(tag_list(train_document))
	trainer = hmm.HiddenMarkovModelTrainer()
	tagger = trainer.train_supervised(train_document)

	print tagger

	# in file tagging
	with codecs.open('./dataset/in.txt','r',encoding='cp720') as f:
		for line in f:
			tagged_seq = tagger.tag(line.strip().split())
			sents_tagged_out_file.write(' '.join('(%s, %s)' % x for x in tagged_seq))
			sents_tagged_out_file.write('\n')


	test_file_path = './dataset/POSteutf.txt'
	test_document = construct_label_sequence(test_file_path)

	predicted_tagged_list = []
	for sent in test_document:
		predicted_tagged_list.extend(sent_tag_list(tagger.tag(sent_word_list(sent))))

	ref = tag_list(test_document)
	print 'test tagset_______'
	print set(ref)

	print 'predicted tagset_______'
	print set(predicted_tagged_list)

	# print 'chito'
	# print ref

	cm = ConfusionMatrix(ref, predicted_tagged_list)
	print cm
	# print cm.pretty_format(show_percents=True, values_in_chart=True,
 #           truncate=None, sort_by_count=False)
	print len(predicted_tagged_list)
	print len(set(predicted_tagged_list))
	print len(ref)
	print len(set(ref))

	print_cm_accuracy(cm)

	normalized_cm = get_normalized_cm(cm)

	# print normalized_cm[1][1]
	l = np.array(normalized_cm)
	cm_out_file.write(np.array2string(l, max_line_width = 200, separator=', '))


	tagger.test(test_document)






