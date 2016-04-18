import os
import random
from math import floor

import nltk

from learning.classifiers.classifier import Classifier
from tree.genus.animal_node import AnimalNode


class StatementClassifier(Classifier):

    genus_direct = 'genus_direct'
    genus_direct_not = 'genus_direct_not'

    def __init__(self):
        super().__init__()

    def train(self, genus_direct_file, genus_direct_not_file):
        # Ensure files exist
        if not os.path.exists(genus_direct_file):
            print("could not find {}".format(genus_direct_file))
            return
        if not os.path.exists(genus_direct_not_file):
            print("could not find {}".format(genus_direct_not_file))
            return

        print("Training statement classifier...")

        genus_direct_content = open(genus_direct_file).read()
        genus_direct_sentences = [sentence for sentence in genus_direct_content.split('\n')]

        genus_direct_not_content = open(genus_direct_not_file).read()
        genus_direct_not_sentences = [sentence for sentence in genus_direct_not_content.split('\n')]

        labeled_names = ([(sentence, self.genus_direct) for sentence in genus_direct_sentences] +
                         [(sentence, self.genus_direct_not) for sentence in genus_direct_not_sentences])
        random.shuffle(labeled_names)

        feature_sets = [(super(StatementClassifier, self)._sentence_features(n), sentiment)
                        for (n, sentiment) in labeled_names]

        self.test_set = feature_sets[:floor(len(feature_sets) / 2)]
        self.training_set = feature_sets[floor(len(feature_sets) / 2):]
        self.classifier = nltk.NaiveBayesClassifier.train(self.training_set)

        print("Statement classifier trained successfully")

    # noinspection PyUnresolvedReferences
    def get_node(self, statement, classification):

        word_tokens = nltk.word_tokenize(statement)
        tagged_words = nltk.pos_tag(word_tokens)

        chunk_grammar = r"""
            genus_direct: {<DT>?<NN><VBZ><DT><NN>}
            """

        chunk_parser = nltk.RegexpParser(chunk_grammar)
        chunks = chunk_parser.parse(tagged_words)

        for chunk in chunks:
            if isinstance(chunk, nltk.tree.Tree):
                if chunk.label() == classification == self.genus_direct:
                    animal_node = AnimalNode()
                    noun_count = 0
                    for child in chunk.leaves():
                        if noun_count == 0 and child[1] == "NN":
                            noun_count = 1
                            animal_node.name = child[0]
                        elif noun_count == 1 and child[1] == "NN":
                            animal_node.genus.append(parent=child[0])
                            return animal_node
