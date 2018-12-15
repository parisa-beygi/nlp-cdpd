from FeatureSelector import *
from HamshahriReader import HamshahriReader
import pickle
import time


# directory = 'test.txt'
# directory = './dataset/HAM2-corpus-short-with-tag-selected.txt'
# FeaturesOutfile = codecs.open('features.txt','w',encoding='cp720')




# print('reading hamshahri...')
# hr = HamshahriReader()
# read_start = time.time()
# hr.read(directory)
# elapsed_read = time.time() - read_start
# hr.print_wordDict()
# hr.print_categories()

start_load = time.time()
with open('resources/hr.pkl', 'rb') as input:
    hr = pickle.load(input)
    print 'loaded hr in ' + str(time.time() - start_load) + ' seconds.\n'

#Testing hr.pkl
# hr.print_wordDict()
# hr.print_categories()

fs_alg = raw_input("choose fs alg:\n 1- X-square\n 2- MutualInfo1\n 3- MutualInfo2\n 4- InfoGain\n 5- MostCommon\n")

agg_policy = 0 if fs_alg in ('4', '5') else raw_input("Max or Sigma?\n 1- Max\n 2- Sigma\n")

if agg_policy is '1':
	sagg = MaxScoreAggregation()
if agg_policy is '2':
	sagg = SigmaScoreAggregation()

if fs_alg is '1':
	fs = XSquareFeatureSelector(hr, sagg)

if fs_alg is '2':
	fs = MutualInfo1FeatureSelector(hr, sagg)

if fs_alg is '3':
	fs = MutualInfo2FeatureSelector(hr, sagg)

if fs_alg is '4':
	fs = InfoGainFeatureSelector(hr)

if fs_alg is '5':
	fs = MostCommonFeatureSelector(hr)

# sagg = MaxScoreAggregation()
# fs = XSquareFeatureSelector(hr, sagg)
fseletion_start = time.time()
features = fs.select_features()
fseletion_elapsed = time.time() - fseletion_start

with open("resources/featuresObj.txt", "wb") as ffs:
	pickle.dump(features, ffs)


print('selected features in ' + str(fseletion_elapsed)) + ' seconds.\n'
