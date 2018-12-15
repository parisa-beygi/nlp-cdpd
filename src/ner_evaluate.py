from __future__ import division
import pickle
import numpy as np
import codecs
import json

CMOutfile = codecs.open('NER_cm.txt','w')

def get_sent_tag_sequence(tagged_sent_list):
	return [tag for word, tag in tagged_sent_list]

def get_NER_dict(tagged_sent_list):
	d = {}
	positive_count = 0

	index = 0
	end = 0
	while index < len(tagged_sent_list):
		word, tag = tagged_sent_list[index]
		if tag != 'O':
			if tag not in d:
				d[tag] = []

			start = index
			end = start + 1
			while end < len(tagged_sent_list):
				w, t = tagged_sent_list[end]
				if t != tag:
					break
				else:
					end += 1
			d[tag].append(tuple((start, end - 1)))
			positive_count += 1

			index = end

		else: # tag == 'O'
			index += 1

	return positive_count, d

def get_match_tag(start, end, tag_seq):
	matched_set = set()
	sub_seq = tag_seq[start: end + 1]
	for t in sub_seq:
		matched_set.add(t)
	if len(matched_set) > 2:
		return 'O'
	elif len(matched_set) == 2:
		if 'O' in matched_set:
			matched_set.remove('O')
			return matched_set.pop()
		else:
			return 'O'
	elif len(matched_set) == 1:
		return matched_set.pop()



def is_type_match(start, end, tag, tag_seq):
	sub_seq = tag_seq[start: end + 1]
	for t in sub_seq:
		if t != tag and t != 'O':
			return False

	return True if tag in sub_seq else False

def is_boundary_match(start, end, tag_seq):
	sub_seq = tag_seq[start: end + 1]

	for t in sub_seq:
		if t == 'O':
			return False

	return True

def is_exact_match(start, end, tag, tag_seq):
	sub_seq = tag_seq[start: end + 1]

	for t in sub_seq:
		if t != tag:
			return False

	return True


def update_cm_entries(cm, cm_entries, tag):

	dim = cm.shape
	n = dim[0]
	updated_cm = cm
	if tag not in cm_entries:
		cm_entries[tag] = n
		updated_cm = np.zeros((n + 1, n + 1))
		updated_cm[:-1, :-1] = cm

	# print '___________'
	# print updated_cm.shape
	return updated_cm, cm_entries



# def get_positive_count(tag_dict):
# 	count = 0

# 	for tag in tag_dict:
# 		count += len(tag_dict[tag])

# 	return count

def update_tag_set(all_test_tag_set, test_dict):
	for x in test_dict:
		all_test_tag_set.add(x)

	return all_test_tag_set

def calc_cm(cm):
	total = 0
	diag = 0
	r, c = cm.shape
	for i in range(r):
		for j in range(c):
			total += cm.item((i, j))

			if i == j:
				diag += cm.item((i, j))

	return 100*(diag/total)



if __name__ == "__main__":

	with open('./resources/test_ner_sent_list.pkl', 'rb') as input:
	    test_sent_list = pickle.load(input)

	with open('./resources/predicted_ner_sent_list.pkl', 'rb') as input:
	    predicted_sent_list = pickle.load(input)

	match = raw_input("1- Type match\n2- Boundary match\n3- Exact match\n")

	tp_count = 0
	test_positive = 0
	predicted_positive = 0


	cm = np.zeros((0,0))
	cm_entries = {}
	all_test_tag_set = set()

	for test_sent, predicted_sent in zip(test_sent_list, predicted_sent_list):
		test_sent_positive, test_dict = get_NER_dict(test_sent)
		predicted_sent_positive, predicted_dict = get_NER_dict(predicted_sent)

		all_test_tag_set = update_tag_set(all_test_tag_set, test_dict)

		test_positive += test_sent_positive
		predicted_positive += predicted_sent_positive

		predicted_sent_tag_seq = get_sent_tag_sequence(predicted_sent)

		for tag in test_dict:

			if match is '1':
				cm, cm_entries = update_cm_entries(cm, cm_entries, tag)

			for start, end in test_dict[tag]:
				if match is '1':
					if is_type_match(start, end, tag, predicted_sent_tag_seq):
						tp_count += 1
						tp = (cm_entries[tag], cm_entries[tag])
						x = cm.item(tp)
						cm.itemset(tp, x + 1)
					else:
						matched = get_match_tag(start, end, predicted_sent_tag_seq)
						# if matched != 'O':
						cm, cm_entries = update_cm_entries(cm, cm_entries, matched)
						tp = (cm_entries[tag], cm_entries[matched])
						x = cm.item(tp)
						cm.itemset(tp, x + 1)

				if match is '2':
					if is_boundary_match(start, end, predicted_sent_tag_seq):
						tp_count += 1

				if match is '3':
					if is_exact_match(start, end, tag, predicted_sent_tag_seq):
						tp_count += 1
	
	if match is '1':	
		print 'type match precision: ', 100*(tp_count/predicted_positive) 
		print 'type match recall: ', 100*(tp_count/test_positive) 
		# print cm.shape
		print cm_entries
		# print len(all_test_tag_set)
		CMOutfile.write(np.array2string(cm, separator=', '))

		CMOutfile.write('\n')
		CMOutfile.write(json.dumps(cm_entries))


		print 'accuracy of the confusion matrix (recall): ', calc_cm(cm)
		
	if match is '2':
		print 'boundary match precision: ', 100*(tp_count/predicted_positive) 
		print 'boundary match recall: ', 100*(tp_count/test_positive) 
	if match is '3':
		print 'exact match precision: ', 100*(tp_count/predicted_positive) 
		print 'exact match recall: ', 100*(tp_count/test_positive) 

