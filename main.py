import nltk

from learning.sequencer.input_sequencer import InputSequencer
from ns_builder import NSBuilder
from semantics.node import Node
from semantics.semantic_network import SemanticNetwork

if __name__ == "__main__":

    ns_builder = NSBuilder()
    ns_builder.add("dog has red fur", name="dog", attribute="has red fur")
    ns_builder.add("dog has orange fur", name="dog", attribute="has orange fur")
    ns_builder.add("cat has ten lives", name="cat", attribute="has ten lives")
    ns_builder.build()

    input_parser = InputSequencer()
    input_parser.train()

    test_network = SemanticNetwork()
    test_network.add_node(input_parser.parse_to_node("dog has purple legs"))
    test_network.print()

    # query = "What is dog"
    # query_tokens = nltk.word_tokenize(query)
    #
    # inherited_bys = []
    # for token in query_tokens:
    #     temp_node = test_network.node_dictionary.get(token, None)
    #
    #     if temp_node is not None:
    #         for herited in temp_node.inherited_by:
    #             inherited_bys.append(herited)
    #
    # inherited_bys.sort(reverse=True)
    #
    # print(inherited_bys)











    # qs_builder = QSBuilder()
    # qs_builder.generate_qs_files()
    #
    # input_classifier = InputClassifier()
    # input_classifier.train()
    # input_classifier.print_accuracy()
    # input_classifier.print_important_features(5)
    # input_classifier.to_pickle()
    #
    # for i in range(10):
    #     print("Enter question or statement")
    #
    #     query = raw_input()
    #     query = query.lower()
    #
    #     word_tokens = nltk.word_tokenize(query)
    #     print("Your tagged query = {}".format(nltk.pos_tag(word_tokens)))
    #
    #     print("This query is a {}".format(input_classifier.classify_text(query)))
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
