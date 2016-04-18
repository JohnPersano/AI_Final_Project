import nltk


class Node:

    def __init__(self, name=""):
        self.name = name
        self.attribute_tokens = []
        self.inherited_by = []

    def set_name(self, name):
        self.name = name

    def add_attribute(self, attribute):
        self.attribute_tokens += nltk.word_tokenize(attribute)

    def add_inherited_by(self, parent):
        self.inherited_by.append(parent)

    def print(self):
        print("Node\n\tname = '{}'\n\tattribute_tokens = {}\n\tinherited_by = {}"
              .format(self.name, self.attribute_tokens, self.inherited_by))

    def to_string(self):
        return "Node\n\tname = '{}'\n\tattribute_tokens = {}\n\tinherited_by = {}"\
            .format(self.name, self.attribute_tokens, self.inherited_by)