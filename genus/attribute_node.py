from genus.base_node import BaseNode


class AttributeNode(BaseNode):

    # Modifier 'enums'
    Verb, Adverb, Adjective, Noun = range(4)

    def __init__(self, pos, name):
        self.pos_titles = ["Verb", "Adverb", "Adjective", "Noun"]
        self.name = name
        self.pos = pos
        self.connections = []

    def append_connection(self, connection=""):
        if connection != "":
            self.connections.append(connection)

    def get_name(self):
        return self.name

    def get_pos(self):
        return self.pos

    def print(self):
        print("<AttributeNode\n\tpos = {}\n\tname = '{}'\n\tconnections = {}>"
              .format(self.pos_titles[self.pos], self.name, self.connections))

    def get_type(self):
        """
        Abstract method.
        """
        return BaseNode.Attribute
