import sys


class Node:
    def __init__(self, label=None):
        self.left = None
        self.right = None
        self.label = label

    def __repr__(self):
        return '{label:' + str(self.label) + '}'

    def __str__(self):
        return str(self.label)

    def insert(self, node):
        if self.left is None:
            self.left = node
        elif self.right is None:
            self.right = node
        else:
            sys.stderr.write('Error: Cannot insert. All branches at node are set.\n')
            sys.exit()

    def printout(self, node):
        tree_ = ''
        if node:
            tree_ += str(node.label)
            if self.left:
                tree_ += self.printout(node.left)
            if self.right:
                tree_ += self.printout(node.right)
        return tree_
