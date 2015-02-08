###01/1O/2014###
###Hackerrank; Category - STATS AND ML; Problem - DOCUMENT CLASSIFICATION; Difficulty - MODERATE###
import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

##Getting training data from file to a list
xtrain_data = []
xtrain_labels = [] 
with open("trainingdata.txt") as f:
	n_train = int(f.readline())
	for line in f:
		xtrain_data.append(' '.join(line.split()[1:]))
		xtrain_labels.append(int(line.split()[0]))


#Getting the count matrix for the training data
# count_vec = CountVectorizer()
# xtrain_counts = count_vec.fit_transform(xtrain_data)

##Tfidf conversion
# tfidf_trans = TfidfTransformer()
# xtrain_tfidf = tfidf_trans.fit_transform(xtrain_counts)

tfidf_trans = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word',stop_words='english')
xtrain_tfidf = tfidf_trans.fit_transform(xtrain_data)

##MultinomialNB classifier
clf = MultinomialNB().fit(xtrain_tfidf, xtrain_labels)

##Testing
n_test = int(raw_input())
xtest_data = []
for i in range(n_test):
	xtest_data.append(raw_input())

#xtest_counts = count_vec.transform(xtest_data)
xtest_tfidf = tfidf_trans.transform(xtest_data) ##Change this to xtest_count when using CountVectorizer and TfidfTransformer

##Prediction
pred = clf.predict(xtest_tfidf)
for i in pred:
	print i
