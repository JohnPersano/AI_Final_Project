import nltk

from learning.input.input_parser import InputParser
from semantics.node import Node
from semantics.semantic_network import SemanticNetwork

if __name__ == "__main__":

    node = Node()
    node.set_name("car")
    node.add_attribute("has paint")

    semantic_network = SemanticNetwork()
    semantic_network.add_node(node)
    semantic_network.print()
    exit()

    input_parser = InputParser()
    input_parser.train_sentence("dog has red fur", semantic_network)

    # print(nltk.pos_tag(nltk.word_tokenize("dog has red fur")))
    # print(nltk.pos_tag(nltk.word_tokenize("dog has blue tail")))

    test_network = SemanticNetwork()
    test_network.add_node(input_parser.parse_to_node("dog has a blue tail"))
    test_network.print()

    query = "What is fur"
    query_tokens = nltk.word_tokenize(query)

    inherited_bys = []
    for token in query_tokens:
        temp_node = test_network.node_dictionary.get(token, None)

        if temp_node is not None:
            for herited in temp_node.inherited_by:
                inherited_bys.append(herited)

    inherited_bys.sort(reverse=True)

    print(inherited_bys[0])











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
