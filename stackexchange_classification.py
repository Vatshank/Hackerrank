import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

##Getting training data from file to a list
xtrain_data_questions = []
xtrain_data_excerpts = []

xtrain_labels = [] 
#xtrain_target = []
with open("training.json") as f:
	n_train = int(f.readline())
	for line in f:
		line_json = json.loads(line)
		for key, value in line_json.iteritems():
			if (key == 'topic'):
				xtrain_labels.append(value)
			elif (key == 'excerpt'):
				xtrain_data_questions.append(value)
			else:
				xtrain_data_excerpts.append(value)

xtrain_data_combined =  [' '.join([xtrain_data_questions[i], xtrain_data_excerpts[i]]) for i in range(len(xtrain_data_questions))]


##Tfidf conversion
tfidf_trans = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word',stop_words='english')
xtrain_tfidf = tfidf_trans.fit_transform(xtrain_data_combined)

##MultinomialNB classifier
clf = MultinomialNB().fit(xtrain_tfidf, xtrain_labels)

##Testing
n_test = int(raw_input())
xtest_data = []
for i in range(n_test):
	test_case = json.loads(raw_input())
	xtest_data.append(test_case['question'])

#xtest_counts = count_vec.transform(xtest_data)
xtest_tfidf = tfidf_trans.transform(xtest_data) 

##Prediction
pred = clf.predict(xtest_tfidf)
for i in pred:
	print i
