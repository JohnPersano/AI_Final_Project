import random

import nltk
from nltk.compat import raw_input

from data.animal_node import AnimalNode
from data.base_node import BaseNode
from data.genus import Genus
from nltk.corpus import names

from learning.input_classifier import InputClassifier

if __name__ == "__main__":

    genus = Genus()

    input_classifier = InputClassifier()
    input_classifier.train(questions_file='C:/Users/John/Development/Python/AI_Final_Project/questions.txt',
                           statements_file='C:/Users/John/Development/Python/AI_Final_Project/statements.txt')

    input_classifier.print_accuracy()
    input_classifier.print_important_features(5)

    for i in range(10):
        print("Enter question or statement")

        query = raw_input()
        query = query.lower()

        print(input_classifier.classify_text(query))

    # word_list = nltk.word_tokenize(query)
    # tagged_words = nltk.pos_tag(word_list)
    #
    # chunk_grammar = "Direct: {<NN><VBZ><DT>?<JJ>*<NN>}"
    #
    # chunk_parser = nltk.RegexpParser(chunk_grammar)
    # chunks = chunk_parser.parse(tagged_words)
    #
    # for chunk in chunks:
    #     if isinstance(chunk, nltk.tree.Tree):
    #         if chunk.label() == 'Direct':
    #             animal_node = AnimalNode()
    #             for child in chunk.leaves():
    #                 if child[1] == "NN":
    #                     animal_node.name = child[0]
    #                     chunk.remove(child)
    #                     break
    #
    #             genus.append_attribute(animal_node, chunk)
    #
    # genus.print()
    #
    # print("Enter simple query")
    #
    # query = raw_input()
    # query = query.lower()
    #
    # response = genus.node_dictionary.get(query, None)
    #
    # if response is not None and response.get_type() == BaseNode.Attribute:
    #     print("{} is associated with {}".format(response.name, response.connections))
