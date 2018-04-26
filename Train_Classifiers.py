#!/usr/bin/env python3

# Training the classifiers and then pickling.
# Executing it sucks time. :P

import nltk
import random
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier

from sentiment.vote_classifier import VoteClassifier
from utils import info, pickling


allowed_word_types = ["J", "R", "V"]


def read_corporas(positive="data/corporas/positive.txt", negative="data/corporas/negative.txt"):
    # Defining and Accessing the corporas.
    # In total, approx 10,000 feeds to be trained and tested on.
    all_words: list = []
    documents: list = []

    info("Accessing the corporas...")

    for p in open(positive, "r"):
        p = p.strip()
        documents.append((p, "pos"))
        words = word_tokenize(p)
        pos = nltk.pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    for p in open(negative, "r"):
        documents.append((p, "neg"))
        words = word_tokenize(p)
        pos = nltk.pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    return documents, all_words

    # # Pickling documents.
    # pickling("data/pickles/documents.pickle", documents)


def get_features(all_words: list, length=5000):
    """Calculate the most frequent words as features

    Arguments:
        all_words {list} -- All words of the string.

    Keyword Arguments:
        length {int} -- Length of features (default: {5000})

    Returns:
        list -- features
    """

    return list(nltk.FreqDist(all_words).keys())[:length]


# # Adjusting the feature finding function, using tokenizing by word in the document.


def find_features(document, features):
    """The feature finding function, using tokenizing by word in the document.
    
    Arguments:
        document {str} -- Document
        features {list} -- List of features
    
    Returns:
        [type] -- [description]
    """

    words = word_tokenize(document)
    _features = {w: (w in words) for w in features}
    return features


# # Shuffling
# random.shuffle(featuresets)
# print(len(featuresets))

# # Partitioning the training and the testing sets.
# testing_set = featuresets[10000:]
# training_set = featuresets[:10000]


# # Training and successive pickling of the classifiers.
# # Takes much time. Be patient.
# classifier = nltk.NaiveBayesClassifier.train(training_set)
# print("Original Naive Bayes Algo accuracy percent:",
#       (nltk.classify.accuracy(classifier, testing_set)) * 100)
# classifier.show_most_informative_features(15)


# save_classifier = open("originalnaivebayes5k.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
# print("MNB_classifier accuracy percent:",
#       (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

# save_classifier = open("MNB_classifier5k.pickle", "wb")
# pickle.dump(MNB_classifier, save_classifier)
# save_classifier.close()

# BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
# BernoulliNB_classifier.train(training_set)
# print("BernoulliNB_classifier accuracy percent:",
#       (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

# save_classifier = open("BernoulliNB_classifier5k.pickle", "wb")
# pickle.dump(BernoulliNB_classifier, save_classifier)
# save_classifier.close()

# LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier.train(training_set)
# print("LogisticRegression_classifier accuracy percent:",
#       (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

# save_classifier = open("LogisticRegression_classifier5k.pickle", "wb")
# pickle.dump(LogisticRegression_classifier, save_classifier)
# save_classifier.close()


# LinearSVC_classifier = SklearnClassifier(LinearSVC())
# LinearSVC_classifier.train(training_set)
# print("LinearSVC_classifier accuracy percent:",
#       (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

# save_classifier = open("LinearSVC_classifier5k.pickle", "wb")
# pickle.dump(LinearSVC_classifier, save_classifier)
# save_classifier.close()


# SGDC_classifier = SklearnClassifier(SGDClassifier())
# SGDC_classifier.train(training_set)
# print("SGDClassifier accuracy percent:", nltk.classify.accuracy(
#     SGDC_classifier, testing_set) * 100)

# save_classifier = open("SGDC_classifier5k.pickle", "wb")
# pickle.dump(SGDC_classifier, save_classifier)
# save_classifier.close()

# # Voting classifier.
# # Basically creates a voting mechanism using the above classifiers.
# # Can be thought of as using an average taking system but not exactly.
# voted_classifier = VoteClassifier(
#     classifier,
#     LinearSVC_classifier,
#     MNB_classifier,
#     BernoulliNB_classifier,
#     LogisticRegression_classifier)

# print("voted_classifier accuracy percent:",
#       (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)


if __name__ == '__main__':
    info("Training classifiers. This may take few minutes to finish.")
    documents, all_words = read_corporas()

    info("Getting top 5000 words as features...")
    word_features = get_features(all_words)
    pickling("data/pickles/word_features5k.pickle", word_features)

    info("Tokenizing and finding features for training...")
    featuresets = [(find_features(rev, word_features), category) for (rev, category) in documents]