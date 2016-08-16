from os import listdir
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def unigram(trainReviews, testReviews):
	print 'unigram'
	vectorizer = CountVectorizer(ngram_range=(1, 1))
	unigramTrain = vectorizer.fit_transform(trainReviews)
	unigramTest = vectorizer.transform(testReviews)
	return unigramTrain, unigramTest

def bigram(trainReviews, testReviews):
	print 'bigram'
	vectorizer = CountVectorizer(ngram_range=(1, 2))
	bigramTrain = vectorizer.fit_transform(trainReviews)
	bigramTest = vectorizer.transform(testReviews)
	return bigramTrain, bigramTest

def unigramTfIdf(trainReviews, testReviews):
	print 'unigram tf-idf'
	vectorizer = TfidfVectorizer(ngram_range=(1, 1))
	unigramTfIdfTrain = vectorizer.fit_transform(trainReviews)
	unigramTfIdfTest = vectorizer.transform(testReviews)
	return unigramTfIdfTrain, unigramTfIdfTest

def bigramTfIdf(trainReviews, testReviews):
	print 'bigram tf-idf'
	vectorizer = TfidfVectorizer(ngram_range=(1, 2))
	bigramTfIdfTrain = vectorizer.fit_transform(trainReviews)
	bigramTfIdfTest = vectorizer.transform(testReviews)
	return bigramTfIdfTrain, bigramTfIdfTest


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

	# testing
	posTestReviews, _ = preprocess(posTestDir)
	negTestReviews, _ = preprocess(negTestDir)
	testReviews = posTestReviews + negTestReviews
	testLabels = [1 for review in posTestReviews] + [-1 for review in negTestReviews]

	# unigram
	# train, test = unigram(trainReviews, testReviews)	
	# train, test = unigramTfIdf(trainReviews, testReviews)
	# train, test = bigram(trainReviews, testReviews)
	train, test = bigramTfIdf(trainReviews, testReviews)
		
	# logistic regression
	print 'starting logistic regression'	
	lr = LogisticRegression()
	lr.fit(train, trainLabels)
	algoLabels = lr.predict(test)
	acc = 1 - sum(np.abs(np.array(trainLabels) - np.array(algoLabels)))/2.0/len(algoLabels)
	print acc

	# random forest
	print 'starting random forest'
	rf = RandomForestClassifier()
	rf.fit(train, trainLabels)
	algoLabels = rf.predict(test)
	acc = 1 - sum(np.abs(np.array(trainLabels) - np.array(algoLabels)))/2.0/len(algoLabels)
	print acc

	# # svm
	# print 'starting svm.'
	# clf = SVC()
	# clf.fit(train, trainLabels)
	# algoLabels = clf.predict(test)
	# acc = 1 - sum(np.abs(np.array(trainLabels) - np.array(algoLabels)))/2.0/len(algoLabels)
	# print acc

		

