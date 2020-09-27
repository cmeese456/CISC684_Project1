import random
import copy
from accuracy import measure_accuracy
import sys

def post_pruning(L,K,tree,validation):
    '''
    Perform the post pruning algorithm on a decision tree. L  is the number of
    attempts to create a new tree. K is the maximum number of subtrees to
    be replaced. validation is a pandas dataframe of the validation set.
    '''
    best_tree = tree
    best_accuracy = measure_accuracy(validation, best_tree)
    random.seed(0)  # To ensure test results can be repeated, seed random with a constant value.
    for i in range(0,L):
        tree_edit = copy.deepcopy(tree)
        M = random.randint(1,K)
        for i in range(0,M):
            # Get a list of all non-leaf nodes in tree_edit. Pick one of these
            # nodes at random and replace it with the most common value among
            # its leafs.
            nonleafs = []
            nonleafs = get_nonleafs(tree_edit)
            N = len(nonleafs)
            #P = random.randint(1,N)
            P = random.randint(0,N-1)
            replace_subtree = nonleafs[P]
            leaf_value = get_majority_class(replace_subtree)
            replace_subtree.label = leaf_value
            replace_subtree.left = None
            replace_subtree.right = None
        current_edit_accuracy = measure_accuracy(validation, tree_edit)
        if (current_edit_accuracy > best_accuracy):
            best_tree = tree_edit
            best_accuracy = current_edit_accuracy
    return best_tree

def get_nonleafs(tree,nonleafs=[]):
    '''
    Return a list of all non-leaf nodes in a tree. Ignore redundant "in-between"
    nodes labeled (as "0" or "1") that denote the direction between an attribute
    and its value.
    '''
    if tree.left or tree.right:
        nonleafs.append(tree)
        if tree.left.left.left or tree.left.left.right:
            get_nonleafs(tree.left.left,nonleafs)
        if tree.right.left.left or tree.right.left.right:
            get_nonleafs(tree.right.left,nonleafs)
    return nonleafs

def get_classifications(tree,classifications=[]):
    '''
    Get all the leaf values (i.e. classifications) in a tree.
    '''
    if tree.left or tree.right:
        '''
        try:
            if tree.left.left or tree.right.left:
                pass
        except AttributeError:
            print(tree.label)
            sys.exit()
        '''
        if tree.left.left.left or tree.left.left.right:
            get_classifications(tree.left.left,classifications)
        else:
            classifications.append(tree.left.left.label)
        if tree.right.left.left or tree.right.left.right:
            get_classifications(tree.right.left,classifications)
        else:
            classifications.append(tree.right.left.label)
    return classifications

def get_majority_class(replace_subtree):
    '''
    Get the most common classification found in the leafs of a tree. If there is
    a tie between two values, return the first value encountered between them.
    '''
    classifications = get_classifications(replace_subtree)
    majority_class = max(classifications,key=classifications.count)
    return majority_class
