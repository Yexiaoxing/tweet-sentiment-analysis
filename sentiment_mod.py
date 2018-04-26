#Run this file after Train_Classifiers.py, where training and pickling happens.
#Or you can also use the saved pickles for running this file, as provided in this repository.
#Creating the sentiment analysis module.
#File: sentiment_mod.py

import pickle
import random
from statistics import mode

import nltk
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC, NuSVC

from vote_classifier import VoteClassifier


classifiers = {
    "NLTK Naive Bayes": "pickles/originalnaivebayes5k.pickle",
    "Multinomial Naive Bayes": "pickles/MNB_classifier5k.pickle",
    "Bernoulli Naive Bayes": "pickles/BernoulliNB_classifier5k.pickle",
    "Logistic Regression": "pickles/LogisticRegression_classifier5k.pickle",
    "SGDClassifier": "pickles/SGDC_classifier5k.pickle",
    "LinearSVC": "pickles/LinearSVC_classifier5k.pickle"
}

trained_classifiers = []
for classifier in classifiers.values():
    with open(classifier, "rb") as fh:
        trained_classifiers.append(pickle.load(fh))

voted_classifier = VoteClassifier(*trained_classifiers)

word_features5k_f = open("pickles/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# Sentiment function only takes one parameter text.
# From there, we break down the features with the find_features function.
def sentiment(text):
    """Sentiment function.
    
    Arguments:
        text {str} -- Tweet string.
    
    Returns:
        str -- sentiment mode (pos or neg)
        int -- confidence
    """

    feats = find_features(text)
    return voted_classifier.sentiment(feats)
