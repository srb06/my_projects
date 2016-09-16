from nltk.corpus import names #Nltk inbuilt corpus/names library to train Naive Bayes
import nltk
import random


# Creating Features - First Name, Last Name, word counts and availability
def gender_features(input_name):
	features = {}
	features["first_letter"] = input_name[0].lower()
	features["last_letter"] = input_name[-1].lower()
	for letter in input_name:
		features["count({})".format(letter)] = input_name.lower().count(letter)
        features["has({})".format(letter)] = (letter in input_name.lower())
	
	return features


##training the classifier using Naive bayes

#Creating list of labeled names
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +[(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)

#Feature Creations for training the data
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

#Dividing the data into training and test set
train_set, test_set = featuresets[500:], featuresets[:500]

#Naive Bayes Classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

#User input of first NAME IN LOWER CASE
input_name = raw_input("Enter first name in Lower case: \n")
feat= gender_features(input_name)
#print(nltk.classify.accuracy(classifier, test_set))
print(classifier.classify(gender_features(input_name)))
