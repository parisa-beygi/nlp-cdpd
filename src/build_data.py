from __future__ import division
from collections import Counter
from sklearn.utils import shuffle
import pickle
import codecs
import time


samplesOutfile = codecs.open('samples.txt','w',encoding='cp720')

# n = int(raw_input('number of features (<100)'))
start_feat = time.time()
with open("resources/featuresObj.txt", "rb") as ffs:
	features = pickle.load(ffs)
# print str(time.time() - start_feat)
# print 'feat pkl time ^'

# features = features[0:n]

start_load = time.time()
with open('resources/hr.pkl', 'rb') as input:
    hr = pickle.load(input)
# print str(time.time() - start_load)
# print 'pkl time ^'

start_build = time.time()
classes = Counter()
i = 0
for cat, count in hr.categories.most_common():
	i += 1
	classes[cat] = i


samples = []
labels = []
for d in hr.documents:
	# d.printDoc()

	sample = []
	summ = 0
	for f in features:
		count = d.dictionary[f]
		sample.append(count)
		summ += count

	if summ != 0:
		sample = [x/summ for x in sample]

#	Sample checking
	for s in sample:
		samplesOutfile.write(str(s) + ' ')

	# samplesOutfile.write(d.domain + '\n')
	samplesOutfile.write(str(classes[d.domain]) + '\n\n')

		
	samples.append(sample)

	labels.append(classes[d.domain])

samples, labels = shuffle(samples, labels)

with open("./resources/dataSamples.txt", "wb") as sampleFile:
	pickle.dump(samples, sampleFile)

with open("./resources/dataLabels.txt", "wb") as labelFile:
	pickle.dump(labels, labelFile)
 
print 'built data in ' + str(time.time() - start_build) + ' seconds.\n'