from nltk.tag import StanfordNERTagger
import nltk.data
import codecs
import pickle
import time



tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def get_tagged_sequence(test_file_path):
	tagged_bodies = []
	body = []
	body_sizes = []

	words_count = 0
	with codecs.open(test_file_path,'r',encoding='cp720') as f:
		for line in f:
			words_count += 1
			contents = line.strip().split()
			if len(contents) != 0:
				word = ' '.join(contents[0:-1])
				body.append(tuple((word, contents[-1])))
			else:
				if body:
					tagged_bodies.append(body)
					body_sizes.append(len(body))
				body = []
		print words_count

		if len(contents) != 0 and body:
			tagged_bodies.append(body)
			body_sizes.append(len(body))


	print len(body_sizes) #1052
	print body_sizes

	return tagged_bodies

def word_list(tagged_body):
	return [word for (word, tag) in tagged_body]

def split_sentence_list(sent_list):
	return [sent.split() for sent in sent_list]


def get_tagged_sents(tagged_bodies):
	# returns [[(word, tag)]: sent]
	tagged_sents_list = []

	for tagged_body in tagged_bodies:
		body_word_list = word_list(tagged_body)
		body_sent_list = tokenizer.tokenize(str(' '.join(body_word_list))) # ['sent']
		sent_list_list = split_sentence_list(body_sent_list) # [[sent]]

		st = 0
		for sent_list in sent_list_list:
			end = st + len(sent_list)
			tagged_sents_list.append(tagged_body[st: end])
			st = end

	return tagged_sents_list



if __name__ == "__main__":
	test_file_path = './dataset/NERte.txt'

	tagged_bodies = get_tagged_sequence(test_file_path)
	tagged_sentences_list = get_tagged_sents(tagged_bodies)
	print 'sents:'
	print tagged_sentences_list[0]
	print tagged_sentences_list[3]





	jar_path = './stanford-ner-tagger/stanford-ner.jar'
	model_path = './stanford-ner-tagger/english-NERtr-model.ser.gz'

	st = StanfordNERTagger(model_path, jar_path)



	begin = time.time()

	predicted_tagged_sent_list = []
	for tagged_sent in tagged_sentences_list:
		sent = word_list(tagged_sent)
		predicted_tagged_sent_list.append(st.tag(sent))

	print 'predicted:'
	print predicted_tagged_sent_list[0]
	print predicted_tagged_sent_list[3]



	# predicted_tagged_corpus = []
	# for tagged_body in tagged_bodies:
	# 	body = word_list(tagged_body)
	# 	sentences_list = tokenizer.tokenize(str(' '.join(body)))
	# 	sent_list_splitted = split_sentence_list(sentences_list)
	# 	predicted_tagged_body = st.tag_sents(sent_list_splitted)
	# 	predicted_tagged_corpus.append(predicted_tagged_body)


	# print predicted_tagged_corpus[0]
	# print predicted_tagged_corpus[-1]

	print time.time() - begin, ' seconds!' 


with open('./resources/test_ner_sent_list.pkl', 'wb') as output:
	pickle.dump(tagged_sentences_list, output, pickle.HIGHEST_PROTOCOL)


with open('./resources/predicted_ner_sent_list.pkl', 'wb') as output:
	pickle.dump(predicted_tagged_sent_list, output, pickle.HIGHEST_PROTOCOL)
