import nltk

from semantics.node import Node


class RelationNode(Node):
    type = "RelationNode"

    def get_type(self):
        return self.type

    def __init__(self):
        super().__init__(self.type)
        self._key = ""
        self._value = ""
        self.value_tokens = []
        self._sentiment = 0.0
        self.in_objects = []
        self.out_objects = []
        self.similar_relations = []

    def set_value(self, value):
        value_tokens = nltk.word_tokenize(value)
        self.value_tokens.clear()
        self.value_tokens += value_tokens
        self._key = self.create_key(value_tokens)
        self._value = value

        # TODO add semtiment

    def add_in_object(self, in_object=None):
        if in_object is None:
            return
        in_object_value = in_object.get_value()
        if not self.__object_exists(self.in_objects, in_object_value):
            self.in_objects.append(in_object_value)

    def add_out_object(self, out_object=None):
        if out_object is None:
            return
        out_object_value = out_object.get_value()
        if not self.__object_exists(self.out_objects, out_object_value):
            self.out_objects.append(out_object_value)

    def get_sentiment(self):
        return self._sentiment

    def get_key(self):
        return self._key

    def get_value(self):
        return self._value

    def print(self):
        print(
            "\n\t\tRelationNode\n\t\t\tname = '{}'\n\t\t\tsentiment = {}"
            "\n\t\t\tin_objects = {}\n\t\t\tout_relationships = {}"
            "\n\t\t\tsimilar_relations = {}\n\t\t"
            .format(self._value, self._sentiment,
                    self.in_objects, self.out_objects,
                    self.similar_relations))

    def to_string(self):
        return "\n\t\tRelationNode\n\t\t\tname = '{}'\n\t\t\tsentiment = {}" \
            "\n\t\t\tin_objects = {}\n\t\t\tout_relationships = {}" \
            "\n\t\t\tsimilar_relations = {}\n\t\t" \
            .format(self._value, self._sentiment,
                    self.in_objects, self.out_objects,
                    self.similar_relations)

    @staticmethod
    def __object_exists(value_list, value):
        for item in value_list:
            if item == value:
                return True
        return False

    @staticmethod
    def create_key(values):
        return "-".join(values)
