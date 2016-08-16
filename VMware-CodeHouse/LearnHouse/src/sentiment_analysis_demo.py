from os import listdir
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def preprocess(directory):
	reviews = list()
	scores = list()
	for fname in listdir(directory):
		reviews.append(open(directory + fname, 'r').read().rstrip().lower())
		scores.append(int(fname[:-4].split('_')[1]))
		# if len(scores) >= 5:
		# 	break

	return reviews, scores


if __name__ == '__main__':

	directory = '/Users/mujq10/Downloads/aclImdb/'
	posTrainDir = directory + 'train/pos/'
	negTrainDir = directory + 'train/neg/'
	posTestDir = directory + 'test/pos/'
	negTestDir = directory + 'test/neg/'
	debug = True

	# training
	posTrainReviews, posScores = preprocess(posTrainDir)
	negTrainReviews, negScores = preprocess(negTrainDir)
	trainReviews = posTrainReviews + negTrainReviews
	trainLabels = [1 for review in posTrainReviews] + [-1 for review in negTrainReviews]

	vectorizer = CountVectorizer(ngram_range=(1, 1))
	train = vectorizer.fit_transform(trainReviews)

	lr = LogisticRegression()
	lr.fit(train, trainLabels)
	while True:
		review = raw_input('input review (EXIT to quit): ')
		if review == 'EXIT':
			break
		test = vectorizer.transform([review])
		algoLabels = lr.predict(test)[0]
		if algoLabels == 1:
			print 'prediction: positive'
		else:
			print 'prediction: negative'