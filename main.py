import nltk
from nltk.compat import raw_input

from genus.animal_node import AnimalNode
from genus.genus import Genus

from learning.input_classifier import InputClassifier
from learning.qs_builder import QSBuilder
from learning.statement_classifier import StatementClassifier

if __name__ == "__main__":

    qs_builder = QSBuilder()
    qs_builder.generate_qs_files()

    input_classifier = InputClassifier()
    input_classifier.train()
    input_classifier.print_accuracy()
    input_classifier.print_important_features(5)
    input_classifier.to_pickle()

    for i in range(10):
        print("Enter question or statement")

        query = raw_input()
        query = query.lower()

        word_tokens = nltk.word_tokenize(query)
        print("Your tagged query = {}".format(nltk.pos_tag(word_tokens)))

        print("This query is a {}".format(input_classifier.classify_text(query)))
    #
    #     if input_classifier.classify_text(query) == 'question':
    #         print("I can't answer questions yet")
    #         continue
    #
    #     print("This type of query is {}".format(statement_classifier.classify_text(query)))
    #     if statement_classifier.classify_text(query) == 'genus_direct_not':
    #         print("I don't know how to handle not statements yet")
    #         continue
    #
    #     print("Resulting genus:")
    #
    #     genus.append_animal(statement_classifier.get_node(query, statement_classifier.classify_text(query)))
    #     genus.print()
