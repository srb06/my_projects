import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import *
import csv
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd

#Creating a list out of the training data
query_list=[]
labels=[]
with open('C:/Users/saurabh/Desktop/bountyApp/train_setsmall.csv', 'rb') as f:
    reader = csv.reader(f)
    reader.next()
    for row in reader:
    	query_list.append(row[1])
    	labels.append(row[2])

#Creating a list out of the final data on which the prediction has to be made
test_list = []
final_labels = []
with open('C:/Users/saurabh/Desktop/bountyApp/oosample_test.csv', 'rb') as fb:
    reader = csv.reader(fb)
    reader.next()
    for row in reader:
    	test_list.append(row[1])
    	final_labels.append(row[2])

stemmer = PorterStemmer()

#Stemming
def stem_tokens(tokens, stemmer):
	stemmed = []
	
	for i in range(len(tokens)):
		#for item in tokens[i]:
			#stemmed.append(stemmer.stem(item))
		stemmed.append([stemmer.stem(item) for item in tokens[i]])

	return stemmed

#Convertin strings into lower case and remove punctuations
def lowerandpunch(query_list):
	query_list = [item.lower() for item in query_list]
	query_list_no_punctuation = [item.translate (None, string.punctuation) for item in query_list]
	return query_list_no_punctuation

tokens = lowerandpunch(query_list)
tokens = [nltk.word_tokenize(item) for item in tokens]

tokens = stem_tokens(tokens,stemmer)

tokens_pred = lowerandpunch(test_list)
tokens_pred = [nltk.word_tokenize(item) for item in tokens_pred]
tokens_pred = stem_tokens(tokens_pred,stemmer)

#dividing into training and testing
train_data,test_data = tokens[:3460], tokens[3460:]
labels_train,labels_test = labels[:3460], labels[3460:]

#TF-IDF Vectorization
tokenize = lambda doc: doc
tfidf = TfidfVectorizer(tokenizer=tokenize,lowercase=False,analyzer='word', min_df = 0,stop_words = 'english')

tfidf_matrix_train = tfidf.fit_transform(train_data)

feature_names = tfidf.get_feature_names()

clf = MultinomialNB().fit(tfidf_matrix_train, labels_train)

# Test sample accuracy

tfidf_matrix_test= tfidf.transform(test_data)
predicted = clf.predict(tfidf_matrix_test)

m=0
for n in range(len(labels_test)):
	if predicted[n]==labels_test[n]:
		m=m+1

print float(m)/float(n)*100


# out of sample Sample Accuracy

tfidf_matrix_outofsample= tfidf.transform(tokens_pred)
predicted = clf.predict(tfidf_matrix_outofsample)

j=0
for i in range(len(final_labels)):
	if predicted[i]==final_labels[i]:
		#print i
		j=j+1
print float(j)/float(i)*100


results = pd.DataFrame({'Final label' : final_labels,
 'predicted' : predicted,
  })

results.to_csv('C:/Users/saurabh/Desktop/bountyApp/final.csv', sep=",")
