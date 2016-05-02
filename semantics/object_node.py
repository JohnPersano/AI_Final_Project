import nltk

from semantics.node import Node
from semantics.relation_node import RelationNode


class ObjectNode(Node):
    type = "ObjectNode"

    def get_type(self):
        return self.type

    class Factory:
        def __init__(self):
            self.value_tokens = []
            self.relations = []
            self.relation_objects = []
            self.sn_classifier = None

        def set_sn_classifier(self, sn_classifier):
            self.sn_classifier = sn_classifier

        def add_value(self, value):
            self.value_tokens.append(value)

        def add_relation(self, out_relation):
            self.relations.append(out_relation)

        def add_relation_object(self, out_object):
            self.relation_objects.append(out_object)

        def build(self):
            node = ObjectNode()
            value = " ".join(self.value_tokens)
            node.set_value(value)

            if len(self.relations) >= 0 and len(self.relation_objects) > 0:
                relationship = RelationNode()
                relationship_value = " ".join(self.relations)
                print("\n\n\n")
                print(relationship_value)
                relationship.set_value(relationship_value, self.sn_classifier)

                relationship_object = ObjectNode()
                relationship_object_value = " ".join(self.relation_objects)
                relationship_object.set_value(relationship_object_value)

                # $node (is a) (cool thing)
                node.add_out_relationship((relationship, relationship_object))
            return node

    def __init__(self):
        super().__init__(self.type)
        self._key = ""               # "red-dog"
        self.value = ""             # "red dog"
        self.value_tokens = []       # [red, dog]
        self.constituents_of = []    # for red node = ["red dog"]
        self.in_relationships = []   # [("dog", "is a"]
        self.out_relationships = []  # [("is a", "color")]

    def __add__(self, other):
        for in_relationship in other.in_relationships:
            if not self.__relationship_exists(self.in_relationships, in_relationship):
                self.in_relationships.append(in_relationship)
        for out_relationship in other.out_relationships:
            if not self.__relationship_exists(self.out_relationships, out_relationship):
                self.out_relationships.append(out_relationship)
        return self

    def set_value(self, value):
        value_tokens = nltk.word_tokenize(value)
        self.value_tokens.clear()
        self.value_tokens += value_tokens
        self._key = "-".join(self.value_tokens)
        self.value = value

    # TODO DOC inherited_by equivalent
    def add_in_relationship(self, relationship_tuple=None):
        if relationship_tuple is None:
            return

        # Get string value for lists e.g. ('dog', 'has a')
        in_object_value = relationship_tuple[0].get_value()
        in_relation_value = relationship_tuple[1].get_value()
        value_tuple = (in_object_value, in_relation_value)

        if not self.__relationship_exists(self.in_relationships, value_tuple):
            self.in_relationships.append(value_tuple)

    def add_out_relationship(self, relationship_tuple=None):
        if relationship_tuple is None:
            return

        # Get string value for lists e.g. ('has a', 'color')
        out_relation_value = relationship_tuple[0].get_value()
        out_object_value = relationship_tuple[1].get_value()
        value_tuple = (out_relation_value, out_object_value)

        if not self.__relationship_exists(self.out_relationships, value_tuple):
            self.out_relationships.append(value_tuple)

    def add_constituent_of(self, constituent):
        if constituent is None:
            return
        constituent_value = constituent.get_value()
        if not self.__constituent_of_exists(self.constituents_of, constituent_value):
            self.constituents_of.append(constituent_value)

    def get_key(self):
        return self._key

    def get_value(self):
        return self.value

    def print(self):
        print(
            "\n\t\tObjectNode\n\t\t\tname = '{}'\n\t\t\tconstituents_of = {}"
            "\n\t\t\tin_relationships = {}\n\t\t\tout_relationships = {}\n\t\t"
            .format(self.value, self.constituents_of,
                    self.in_relationships, self.out_relationships))

    def to_string(self):
        return "\n\t\tObjectNode\n\t\t\tname = '{}'\n\t\t\tconstituents_of = {}" \
                "\n\t\t\tin_relationships = {}\n\t\t\tout_relationships = {}\n\t\t" \
                .format(self.value, self.constituents_of,
                        self.in_relationships, self.out_relationships)

    @staticmethod
    def __relationship_exists(relationships_list, value_tuple):
        for item in relationships_list:
            if item == value_tuple:
                return True
        return False

    @staticmethod
    def __constituent_of_exists(constituents_list, constituent_value):
        for item in constituents_list:
            if item == constituent_value:
                return True
        return False
