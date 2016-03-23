import nltk
from nltk.compat import raw_input
from nltk.corpus import stopwords

from tree.node import Node
from tree.tree import Tree

if __name__ == "__main__":

    tree = Tree()

    for i in range(2):
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
                    node = Node()
                    node_set = False
                    for child in chunk.leaves():
                        if child[1] == "NN":
                            if node_set is False:
                                node.name = child[0]
                                node_set = True
                            else:
                                node.parent = child[0]
                    tree.append(node)

    tree.print()



    # print(result)

    # # Creates a new tree with mammal as the only node
    # tree = Tree()
    #
    # # There are a bunch of different ways to add nodes, I tried to make it flexible
    #
    # # Simplest way
    # dog = Node("dog")
    # tree.append(dog)
    #
    # # Dog has parent 'mammal', let's change dogs parent to canine
    # canine = Node("canine")
    # canine.append(child=dog.name)
    # tree.append(canine)
    #
    # # Dog now has parent canine, let's change it back to mammal a different way
    # dog = Node("dog")  # Refresh reference
    # dog.append(parent="mammal")
    # tree.append(dog)
    #
    # # Mammal now has two children, dog and canine, none of which have their own children. Let's add a cat and lion
    # lion = Node("lion")
    # lion.append(parent="cat")
    # tree.append(lion)
    #
    # # Let's fix dog to have parent canine again
    # canine = Node("canine")
    # canine.append(child="dog")
    # tree.append(canine)
    #
    # # Show the individual nodes in the dictionary
    # tree.print()
    #
    # """
    #     You will get the following structure at the end:
    #
    #     Mammal:
    #         Canine:
    #             Dog:
    #         Cat:
    #             Lion:
    # """
    # filtered_words = [word for word in word_list if word not in stopwords.words('english')]
