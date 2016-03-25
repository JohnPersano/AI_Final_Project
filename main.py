import nltk
from nltk.compat import raw_input

from data.animal_node import AnimalNode
from data.base_node import BaseNode
from data.genus import Genus

if __name__ == "__main__":

        genus = Genus()

        animal = AnimalNode("canine")
        animal.genus.append(child="dog")
        genus.append_animal(animal)

        print("Enter information")

        question = raw_input()
        question = question.lower()

        word_list = nltk.word_tokenize(question)
        tagged_words = nltk.pos_tag(word_list)

        chunk_grammar = "Direct: {<NN><VBZ><DT>?<JJ>*<NN>}"

        chunk_parser = nltk.RegexpParser(chunk_grammar)
        chunks = chunk_parser.parse(tagged_words)

        for chunk in chunks:
            if isinstance(chunk, nltk.tree.Tree):
                if chunk.label() == 'Direct':
                    animal_node = AnimalNode()
                    for child in chunk.leaves():
                        if child[1] == "NN":
                            animal_node.name = child[0]
                            chunk.remove(child)
                            break

                    genus.append_attribute(animal_node, chunk)

        genus.print()

        print("Enter simple query")

        query = raw_input()
        query = query.lower()

        response = genus.node_dictionary.get(query, None)

        if response is not None and response.get_type() == BaseNode.Attribute:
            print("{} is associated with {}".format(response.name, response.connections))


    # tree = Tree()
    #
    # for i in range(2):
    #     print("Enter information")
    #
    #     question = raw_input()
    #     question = question.lower()
    #
    #     word_list = nltk.word_tokenize(question)
    #     tagged_words = nltk.pos_tag(word_list)
    #
    #     chunk_grammar = "Direct: {<NN><VBZ><DT>?<JJ>*<NN>}"
    #
    #     chunk_parser = nltk.RegexpParser(chunk_grammar)
    #     chunks = chunk_parser.parse(tagged_words)
    #
    #     for chunk in chunks:
    #         if isinstance(chunk, nltk.tree.Tree):
    #             if chunk.label() == 'Direct':
    #                 node = Node()
    #                 node_set = False
    #                 for child in chunk.leaves():
    #                     if child[1] == "NN":
    #                         if node_set is False:
    #                             node.name = child[0]
    #                             node_set = True
    #                         else:
    #                             node.parent = child[0]
    #                 tree.append(node)
    #
    # tree.print()

