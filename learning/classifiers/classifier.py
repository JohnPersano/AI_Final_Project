"""
CSCI 6660 Final Project

Author: John Persano
Date:   04/04/2016
"""

from collections import Counter

import nltk


class Classifier:
    """
    The Classifier class serves as a base class for the SNClassifier and QSClassifier. This Classifier uses
    a Naive Bayes classifier with various features.
    """
    def __init__(self):
        self.classifier = None
        self.test_set = []
        self.training_set = []

    def train(self, *args):
        raise NotImplementedError("All classes implementing Classifier should implement train()")

    @staticmethod
    def _sentence_features(sentence):
        """
        Gets a feature dictionary for a particular sentence to use with the NaiveBayesClassifier.
        :param sentence: a sentence to 'featurize'
        :return: a feature dictionary for the sentence
        """
        features = {}

        word_tokens = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(word_tokens)

        # May cause an error is rare situations
        if not tagged_words:
            return features

        # This is a bag of words feature
        word_tokens = nltk.word_tokenize(sentence)
        for word in word_tokens:
            features[word] = True

        # This is a bi-gram feature
        for bigram in nltk.bigrams(word_tokens):
            features[bigram] = True

        """
        The below feature is commented out because it was shown to have a negative
        impact on sentiment classification
        """
        # This is a tag feature
        # for tag in tagged_words:
        #     try:
        #         features[tag[1]] = True
        #     except IndexError:
        #         return features

        # This is pos count feature
        features.update(Counter(tag for word, tag in tagged_words))

        return features

    def print_accuracy(self):
        print("Classifier accuracy: {}%"
              .format(int(nltk.classify.accuracy(self.classifier, self.test_set) * 100)))

    def print_important_features(self, amount=15):
        print(self.classifier.show_most_informative_features(amount))

    def classify_text(self, query=""):
        """
        Classifies the query into one of the specified classes.
        """
        return self.classifier.classify(self._sentence_features(query))

    def prob_classify_text(self, query=""):
        """
        Classifies the query into one of the specified classes and returns the probability
        that it is correct.
        """
        probabilities = self.classifier.prob_classify(self._sentence_features(query))
        for sample in probabilities.samples():
            return sample, probabilities.prob(sample)
