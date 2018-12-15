from nltk.tag import StanfordNERTagger
import nltk.data
import codecs
import time



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

if __name__ == "__main__":
	test_file_path = ./dataset/NERte.txt'

	tagged_bodies = get_tagged_sequence(test_file_path)

	# tagged_bodies = tagged_bodies[66]
	# print tagged_bodies

	jar_path = './stanford-ner-tagger/stanford-ner.jar'
	model_path = './stanford-ner-tagger/english-NERtr-model.ser.gz'

	st = StanfordNERTagger(model_path, jar_path)
	# l = body_word_list(tagged_bodies)
	# print 'word list:'
	# print l

	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	# print 'l:____________', type(l)
	# text = str(' '.join(l))
	# print text
	# sent_list = tokenizer.tokenize(text)

	# print 'sentence list___________'
	# print sent_list

	# a = [sent.split() for sent in sent_list]

	# print 'aaaaaaaaaaaa'
	# print a
	# tagged_sents = st.tag_sents(a)
	# print tagged_sents

	begin = time.time()

	predicted_tagged_corpus = []
	for tagged_body in tagged_bodies:
		body = word_list(tagged_body)
		sentences_list = tokenizer.tokenize(str(' '.join(body)))
		sent_list_splitted = split_sentence_list(sentences_list)
		predicted_tagged_body = st.tag_sents(sent_list_splitted)
		predicted_tagged_corpus.append(predicted_tagged_body)


	print predicted_tagged_corpus[0]
	print predicted_tagged_corpus[-1]

	print time.time() - begin, ' seconds!' 
