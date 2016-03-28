# Node class
import os
import random

import re

import nltk
from math import floor
from nltk.corpus import names


class InputClassifier:

    def __init__(self):
        self.classifier = None
        self.test_set = []
        self.training_set = []

    def train(self, questions_file="", statements_file=""):
        # Ensure files exist
        if not os.path.exists(questions_file) and os.path.exists(statements_file):
            if not os.path.exists(questions_file):
                print("could not find {}".format(questions_file))
            else:
                print("could not find {}".format(statements_file))
            return

        questions = [re.sub(r'[^\w\s]', '', name.lower()) for name in names.words(questions_file)]
        statements = [re.sub(r'[^\w\s]', '', name.lower()) for name in names.words(statements_file)]

        labeled_names = ([(name, 'question') for name in questions] +
                         [(name, 'statement') for name in statements])
        random.shuffle(labeled_names)

        feature_sets = [(self.__word_features(n), gender) for (n, gender) in labeled_names]

        self.test_set = feature_sets[:floor(len(feature_sets) / 2)]
        self.training_set = feature_sets[floor(len(feature_sets) / 2):]
        self.classifier = nltk.NaiveBayesClassifier.train(self.training_set)

        print("Input classifier trained successfully")

    def __word_features(self, words):

        # Add more features here
        return {word: True for word in words}

    def print_accuracy(self):
        print("Input classifier accuracy: {}%"
              .format(int(nltk.classify.accuracy(self.classifier, self.test_set) * 100)))

    def print_important_features(self, amount=15):
        print(self.classifier.show_most_informative_features(amount))

    def classify_text(self, query=""):
        return self.classifier.classify(self.__word_features(query))

