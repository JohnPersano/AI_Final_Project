import nltk


class Node:
    def __init__(self, name=""):
        self.name = name
        self.attribute_tokens = []
        self.attributes = []
        self.inherited_by = []
        self.attribute_string_builder = None

    def set_name(self, name):
        self.name = name

    def add_attribute(self, new_attribute):
        if new_attribute is None:
            return
        if not self.__exists(self.attributes, new_attribute):
            self.attributes.append(new_attribute)
        for word_token in nltk.word_tokenize(new_attribute):
            if not self.__exists(self.attribute_tokens, word_token):
                self.attribute_tokens.append(word_token)

    def build_attribute(self, attribute_token):
        if self.attribute_string_builder is None:
            self.attribute_string_builder = attribute_token
        else:
            self.attribute_string_builder += " "
            self.attribute_string_builder += attribute_token

    def add_inherited_by(self, new_parent):
        if not self.__exists(self.inherited_by, new_parent):
            self.inherited_by.append(new_parent)

    @staticmethod
    def __exists(value_list, value):
        for item in value_list:
            if item == value:
                return True
        return False

    def print(self):
        print(
            "\n\t\tNode\n\t\t\tname = '{}'\n\t\t\tattribute_tokens = {}"
            "\n\t\t\tattributes = {}\n\t\t\tinherited_by = {}\n\t\t"
            .format(self.name, self.attribute_tokens, self.attributes, self.inherited_by))

    def to_string(self):
        return "\n\t\tNode\n\t\t\tname = '{}'\n\t\t\tattribute_tokens = {}" \
               "\n\t\t\tattributes = {}\n\t\t\tinherited_by = {}\n\t\t" \
                .format(self.name, self.attribute_tokens, self.attributes, self.inherited_by)
