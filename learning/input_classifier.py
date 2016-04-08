import os
import random

import nltk
from math import floor

from learning.classifier import Classifier


class InputClassifier(Classifier):

    def __init__(self):
        super().__init__()

    def train(self, questions_file="", statements_file=""):
        # Ensure files exist
        if not os.path.exists(questions_file) and os.path.exists(statements_file):
            if not os.path.exists(questions_file):
                print("could not find {}".format(questions_file))
            else:
                print("could not find {}".format(statements_file))
            return

        print("Training input classifier...")

        questions_content = open(questions_file).read()
        question_sentences = [sentence for sentence in questions_content.split('\n')]

        statements_content = open(statements_file).read()
        statements_sentences = [sentence for sentence in statements_content.split('\n')]

        labeled_names = ([(sentence, 'question') for sentence in question_sentences] +
                         [(sentence, 'statement') for sentence in statements_sentences])
        random.shuffle(labeled_names)

        feature_sets = [(super(InputClassifier, self)._sentence_features(n), sentiment)
                        for (n, sentiment) in labeled_names]

        self.test_set = feature_sets[:floor(len(feature_sets) / 2)]
        self.training_set = feature_sets[floor(len(feature_sets) / 2):]
        self.classifier = nltk.NaiveBayesClassifier.train(self.training_set)

        print("Input classifier trained successfully")
