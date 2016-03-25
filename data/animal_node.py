from data.base_node import BaseNode


class AnimalNode(BaseNode):

    def __init__(self, name=""):
        self.genus = self.Genus()
        self.name = name
        self.attributes = []

    class Genus:

        def __init__(self, parent=""):
            self.children = []
            self.parent = parent

        def append(self, child="", parent=""):
            if child != "":
                self.children.append(child)
            if parent != "":
                self.parent = parent

        def remove_child(self, child):
            if child.name in self.children:
                self.children.remove(child.name)

    def append(self, attribute=""):
        if attribute != "":
            self.attributes.append(attribute)

    def get_name(self):
        return self.name

    def print(self):
        print("<AnimalNode\n\tname = '{}'\n\tgenus_children = {}\n\tgenus_parent = '{}'\n\tattributes = {}>"
              .format(self.name, self.genus.children, self.genus.parent, self.attributes))

    def get_type(self):
        """
        Abstract method.
        """
        return BaseNode.Animal
