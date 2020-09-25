import pandas as pd
import math
import copy
from CISC684_Project1.tree import *
import random
||||||| a3b21ab
import random
=======
import random
from CISC684_Project1.tree import *
>>>>>>> 674915b821c5632699b59476427ea1ad98a6fe85

"""This file is for implementing the ID3 decision tree learning algorithm. As discussed in class,
the main step in decision tree learning is choosing the next attribute to split on. Implement the
following two heuristics for selecting the next attribute.

1. Information Gain heuristic (see Mitchell slides Ch. 3)

2. Variance impurity heuristic (described below)"""

# Below is some code to ensure the program is called with the correct arguments and to grab the input values
# It is commented for now until we need to use it and test it
"""
# Ensure arguments were passed into the application
if(len(sys.argv) != 6)
    print 'Incorrect Usage! Please enter six arguments: <L>, <K>, <training-set>, <validation-set>, <test-set>,
     <to-print>'
    sys.exit(2)

# Get command line arguments
l = sys.argv[0]
k = sys.argv[1]
training_set = sys.argv[2]
validation_set = sys.argv[3]
test_set = sys.argv[4]
to_print = sys.argv[5]
"""
# Pandas Data Frame
df = pd.read_csv('data_sets2/data_sets2/training_set.csv')

'''
    def printout(self, node):
        tree = ''
        if node:
            tree += str(node.label)
            if (self.left):
                tree += self.printout(node.left)
            if (self.right):
                tree += self.printout(node.right)
        return tree
'''


def printy(node, depth=0):
    prefix = "| " * depth
    if node:
        if node.left.left.left or node.left.left.right:
            print(prefix + node.label + " = " + str(node.left.label) + " : ")
            printy(node.left.left, depth+1)
        else:
            print(prefix + node.label + " = " + str(node.left.label) + " : " + str(node.left.left.label))
        if node.right.left.left or node.right.left.right:
            print(prefix + node.label + " = " + str(node.right.label) + " : ")
            printy(node.right.left, depth+1)
        else:
            print(prefix + node.label + " = " + str(node.right.label) + " : " + str(node.right.left.label))
    return


# Get the label (i.e. attribute name) with the highest information gain.
def information_gain_heuristic(s, attr_list, df):
    gain_list = []
    for attr in attr_list:
<<<<<<< HEAD
        gain_list.append(information_gain(s, attr, df))  # failing within information_gain
    return max(gain_list, key=lambda item: item[1])[0]

||||||| a3b21ab
        gain_list.append(information_gain(s, attr, df))
    return max(gain_list, key=lambda item:item[1])[0]
=======
        gain_list.append(information_gain(s, attr, df))
    return max(gain_list, key=lambda item: item[1])[0]

>>>>>>> 674915b821c5632699b59476427ea1ad98a6fe85

# Get the information gain for an attribute.
def information_gain(s, attr, df):
<<<<<<< HEAD
    s_pos = len(df.loc[(df[s] == 1)])   # Number of positive (i.e. Class=1) examples
    s_neg = len(df.loc[(df[s] == 0)])   # Number of negative (i.e. Class=0) examples
    s_total = len(df.index)             # Total number of examples
    attr_val0_pos = 0
    attr_val0_neg = 0
    attr_val1_pos = 0
    attr_val1_neg = 0
    for i in range(len(df[s])):
        try:
            if df.iloc[i].loc[s] == 1 and df.iloc[i].loc[attr] == 0:
                attr_val0_pos += 1
            elif df.iloc[i].loc[s] == 0 and df.iloc[i].loc[attr] == 0:
                attr_val0_neg += 1
            elif df.iloc[i].loc[s] == 1 and df.iloc[i].loc[attr] == 1:
                attr_val1_pos += 1
            elif df.iloc[i].loc[s] == 0 and df.iloc[i].loc[attr] == 1:
                attr_val1_neg += 1
        except KeyError as e:
            print(attr)
            print(i)
            print(df.iloc[i].loc[s])
            # print(df.iloc[i].loc[attr])
            print(df.iloc[i])
            print(df)
            sys.exit()
||||||| a3b21ab
    s_pos = len(df.loc[(df[s] == 1)])
    s_neg = len(df.loc[(df[s] == 0)])
    s_total = len(df.index)
    attr_val0_pos = len(df.loc[(df[s] == 1) & df[attr] == 0])
    attr_val0_neg = len(df.loc[(df[s] == 0) & df[attr] == 0])
=======
    s_pos = len(df.loc[(df[s] == 1)])   # Number of positive (i.e. Class=1) examples
    s_neg = len(df.loc[(df[s] == 0)])   # Number of negative (i.e. Class=0) examples
    s_total = len(df.index)             # Total number of examples
    attr_val0_pos = 0
    attr_val0_neg = 0
    attr_val1_pos = 0
    attr_val1_neg = 0
    for i in range(len(df[s])):
        try:
            #if df[s][i] == 1 and df[attr][i] == 0:
            if df.iloc[i].loc[s] == 1 and df.iloc[i].loc[attr] == 0:
                attr_val0_pos += 1
            #elif df[s][i] == 0 and df[attr][i] == 0:
            elif df.iloc[i].loc[s] == 0 and df.iloc[i].loc[attr] == 0:
                attr_val0_neg += 1
            #elif df[s][i] == 1 and df[attr][i] == 1:
            elif df.iloc[i].loc[s] == 1 and df.iloc[i].loc[attr] == 1:
                attr_val1_pos += 1
            #elif df[s][i] == 0 and df[attr][i] == 1:
            elif df.iloc[i].loc[s] == 0 and df.iloc[i].loc[attr] == 1:
                attr_val1_neg += 1
        except KeyError as e:
            print(attr)
            print(i)
            print(df.iloc[i].loc[s])
            #print(df.iloc[i].loc[attr])
            print(df.iloc[i])
            print(df)
            sys.exit()
>>>>>>> 674915b821c5632699b59476427ea1ad98a6fe85
    attr_val0_total = len(df.loc[(df[attr] == 0)])
    attr_val1_total = len(df.loc[(df[attr] == 1)])
    attr_total = len(df.index)
    # The entropy of the entire set
    entropy_s = get_entropy(s, s_pos, s_neg, s_total)
    # The entropy of the subset that has "0" as a value for the attribute
    try:
        entropy_attr_0 = get_entropy(attr, attr_val0_pos, attr_val0_neg, attr_val0_total)
    except ZeroDivisionError:
        print(attr)
        print(df)
        sys.exit()
    # The entropy of the subset that has "1" as a value for the attribute
    try:
        entropy_attr_1 = get_entropy(attr, attr_val1_pos, attr_val1_neg, attr_val1_total)
    except ZeroDivisionError:
        print(attr)
        print(df)
        sys.exit()
    # Subtract from the set's entropy the entropy of each value multiplied by its proportion in the set
    gain = entropy_s[1] - (attr_val0_total / attr_total) * entropy_attr_0[1] - (attr_val1_total / attr_total) * entropy_attr_1[1]
    # Return the label of the attribute and its gain
    return attr, gain


def get_entropy(node_label, val1_instances, val0_instances, total):
    """The formula for entropy is Entropy = -p_1 * log_2(p_1) - p_0 * log_2(p_0)"""
    """As per Mitchell p. 56, 0 log 0 is defined as zero """
<<<<<<< HEAD
    if total == 0:
        entropy = 0
    else:
        entropy = (-1 * val1_instances / total * math.log2(
            val1_instances / total if val1_instances / total > 0 else 1)) - \
                  (val0_instances / total * math.log2(val0_instances / total if val0_instances / total > 0 else 1))
    return node_label, entropy

||||||| a3b21ab
    try:
        entropy = (-1 * val1_instances/total * math.log2(val1_instances/total if val1_instances/total > 0 else 1)) - \
                  (val0_instances/total * math.log2(val0_instances/total if val0_instances/total > 0 else 1))
    # I'm getting an error where I have more attributes set to 0 or 1 than I
    # have total rows.
    except ZeroDivisionError as e:
        print("val1_instances: " + str(val1_instances))
        print("val0_instances: " + str(val0_instances))
        print("total: " + str(total))
        print("node: " + node_label)
        print(e)
        sys.exit()
    return node_label, entropy

# This is the original get_entropy from Matt's commit
def get_entropy_matt(node_label, positive_instances, negative_instances, total):
    """The formula for entropy is Entropy = -p_1 * log_2(p_1) - p_0 * log_2(p_0)"""
    entropy = (-1 * positive_instances/total * math.log2(positive_instances/total)) - \
              (negative_instances/total * math.log2(negative_instances/total))
    return node_label, entropy

def get_attribute_labels(dataset):
    column_names = dataset.columns

    attribute_labels_list = []
    positive_instances_list = []
    negative_instances_list = []
    total_instances_list = []
    for col in column_names:
        positive_instances = 0
        negative_instances = 0
        current_column = dataset[col]
        for i in range(len(current_column)):
            if current_column[i] == 1:
                positive_instances += 1
            elif current_column[i] == 0:
                negative_instances += 1
            else:
                sys.stderr.write('ILLEGAL VALUE FOUND IN CLASS DATA')
                sys.exit()
        total = positive_instances + negative_instances
        # for each column, append the column label, the number of 1s, the number of 0s, and the total number of values
        attribute_labels_list.append(col)
        positive_instances_list.append(positive_instances)
        negative_instances_list.append(negative_instances)
        total_instances_list.append(total)

    return attribute_labels_list, positive_instances_list, negative_instances_list, total_instances_list
=======
    # If there are no more examples with this value for this attribute, Return
    # entropy = 0. This avoids a divide by zero error.
    if total == 0:
        entropy = 0
    else:
        entropy = (-1 * val1_instances/total * math.log2(val1_instances/total if val1_instances/total > 0 else 1)) - \
              (val0_instances/total * math.log2(val0_instances/total if val0_instances/total > 0 else 1))
    return node_label, entropy

>>>>>>> 674915b821c5632699b59476427ea1ad98a6fe85

# examples_list     A pandas dataframe
# target_attribute  The name of the target attribute (i.e. "Class")
# attributes_list   A list of attribute names (i.e. "XC", "XD", etc.)
def id3(examples_list, target_attribute, attributes_list):
    root = Node()
    A = ''
    # If all examples are positive, set root's label to "1"
    if all(l == 1 for l in list(examples_list[target_attribute])):
        root.label = 1
    # Elif all examples are negative, set root's label to "0"
    elif all(l == 0 for l in list(examples_list[target_attribute])):
        root.label = 0
    # Elif attributes_list is empty, set root's label to most common value of
    #   target attribute in examples
    elif not attributes_list:
        root.label = examples_list[target_attribute].mode()[0]
    else:
        # A = random.choice(attributes_list) # For testing. Swap with info_gain
        A = information_gain_heuristic(target_attribute, attributes_list, examples_list)
        root.label = A
<<<<<<< HEAD
        for i in [0, 1]:
||||||| a3b21ab
        for i in [0,1]:
=======
        for i in [0,1]:
            # The code below creates a node in-between a parent node and its
            # child. The sole purpose of this in-between node is to bear a label
            # reflecting whether the parent node's value was 0 or 1 for the
            # child branch. BUT that information is already conveyed by the
            # order in which nodes are inserted. The first insertion goes to the
            # left and signifies a "0" value for the parent node. The second
            # goes on the right and signifies a "1" value for the parent node.
            # So this in-between node is redundant.
>>>>>>> 674915b821c5632699b59476427ea1ad98a6fe85
            new_branch = Node()
            new_branch.label = str(i)
            examples_list_vi = examples_list.loc[examples_list[A] == i]
            if examples_list_vi.empty:
                new_leaf = Node()
                new_leaf.label = examples_list[target_attribute].mode()[0]
                new_branch.insert(new_leaf)
                root.insert(new_branch)
            else:
                trimmed_attributes = copy.deepcopy(attributes_list)
                trimmed_attributes.remove(A)
                new_branch.insert(id3(examples_list_vi, target_attribute, trimmed_attributes))
                root.insert(new_branch)
            # This code does away with separate "in-between" nodes. The
            # attribute value that leads from a parent to its child is
            # represented by which side the child is inserted on. Left nodes
            # (inserted first) represent a "0" value. Right nodes represent
            # a "1" value.
            '''
            examples_list_vi = examples_list.loc[examples_list[A] == i]
            if (examples_list_vi.empty):
                new_leaf = Node()
                new_leaf.label = examples_list[target_attribute].mode()[0]
                root.insert(new_leaf)
            else:
                trimmed_attributes = copy.deepcopy(attributes_list)
                trimmed_attributes.remove(A)
                root.insert(id3(examples_list_vi,target_attribute,trimmed_attributes))
                '''
    return root

<<<<<<< HEAD

# Note: Variance Impurity Heuristics are based off Evan's work on the Node and Attribute Classes
# They will only work once we merge this code with his side branch
||||||| a3b21ab
#! Note: Variance Impurity Heuristics are based off Evan's work on the Node and Attribute Classes
#! They will only work once we merge this code with his side branch
=======

#! Note: Variance Impurity Heuristics are based off Evan's work on the Node and Attribute Classes
#! They will only work once we merge this code with his side branch
>>>>>>> 674915b821c5632699b59476427ea1ad98a6fe85
# This function calculates the variance impurity of all attributes in attr_dict in comparison to the total set s
# Important: attr_dict must only contain attributes to be tested, prune the input based on tree level before calling
def variance_impurity_heuristic(s, attr_dict):
    # Keep a list of the variance gain of each attribute
    gain_list = []

    # Calculate the variance impurity gain of each attribute in the list
    for attr in attr_dict:
        gain_list.append(variance_impurity_gain(s, attr_dict[attr]))

    # Return the label of the attribute with the highest gain from the list
    return max(gain_list, key=lambda item: item[1])[0]



def variance_impurity_gain(s, attr):
    # Get the variance impurity of the entire set
    variance_s = variance_impurity(s.label, s.getTotalVal1(), s.getTotalVal0(), s.getTotal())

    # Get the variance impurity of the subset that has "0" as a value for the attribute
    variance_attr_0 = variance_impurity(attr.label, attr.val0_pos, attr.val0_neg, attr.getTotalVal0())

    # Get the variance impurity of the subset that has "1" as the value for the attribute
    variance_attr_1 = variance_impurity(attr.label, attr.val1_pos, attr.val1_neg, attr.getTotalVal0())

    # Subtract the weighted variance impurity of each value from that of the total set
    variance_gain = variance_s[1] - ((attr.getTotalVal0() / attr.getTotal()) * variance_attr_0[1] -
                                     (attr.getTotalVal1() / attr.getTotal()) * variance_attr_1[1])
    # Return label and gain
    return attr.label, variance_gain


# Variance Impurity = (K0/K)*(K1/K)
def variance_impurity(node_label, val1_instances, val0_instances, total):
    # Calculate the variance impurity using the formula from the homework
    variance_impurity_ = (val1_instances / total) * (val0_instances / total)

    # Return the impurity and the label
    return node_label, variance_impurity_

<<<<<<< HEAD

def standard_output_format(dt):
||||||| a3b21ab

def standard_output_format():
=======
def standard_output_format(dt):
>>>>>>> 674915b821c5632699b59476427ea1ad98a6fe85
    """This function prints the decision tree to standard output. An example of standard output is below.
    wesley = 0 :
    | honor = 0 :
    | | barclay = 0 : 1
    | | barclay = 1 : 0
    | honor = 1 :
    | | tea = 0 : 0
    | | tea = 1 : 1
    wesley = 1 : 0
    According to this tree, if wesley = 0, honor = 0, and barclay = 0, then the class value of the corresponding
    instance should be 1. In other words, the value appearing before a colon is an attribute value, and the value
    appearing after a colon is a class value."""
<<<<<<< HEAD
    tree_level = 0
    left_node = 0
    right_node = 0
    while left_node is not None and right_node is not None:
        if tree_level == 0:
            if left_node is not None:
                left_node = dt.left
                print(dt.label, '=', left_node.label, ':')
            elif right_node is not None:
                right_node = dt.right
                print(dt.label, '=', right_node.label, ':')
        elif tree_level > 0:
            while left_node.left is not None:
                to_print = ''
                for i in range(tree_level):
                    to_print += '| '
                left_node = left_node.left
                left_branch_value = left_node.left
                left_branch_value = left_branch_value.label
                if type(left_branch_value) != str:
                    str(left_branch_value)
                to_print += left_node.label + ' = ' + left_branch_value + ' : '
                print(to_print)
                tree_level += 1
        tree_level += 1

        # if tree_level > 0:
        #     for i in range(tree_level):
        #         visual_level += '| '
        #     if left_node is not None:
        #         left_node = left_node.left
        #         left_node = left_node.label
        #         visual_level += left_node.label
        #
        #         print(visual_level, '=',  ':', )
        #     elif right_node is not None:
        #         right_node = right_node.right
        #         visual_level += right_node.label
        # if tree_level == 0:  # needed to format tree to project specifications
        #     # print(dt.label, '= ', VALUE, ': ', VALUE)
        #     if dt.left is not None:
        #         left_node = dt.left
        #         left_node = left_node.label  # each branch has a new node set up in the label field
        #         branch_label = left_node.label  # call label again to get the label of the branch, 0 or 1
        #         print(dt.label, '=', branch_label, ': ')
        #     elif dt.right is not None:
        #         right_node = dt.right
        #         right_node = right_node.label
        #         branch_label = right_node.label
        #         print(dt.label, '=', branch_label, ': ')
        # tree_level += 1


||||||| a3b21ab
    return ''


# class_label, pos, neg, tot = get_class_labels(df)
# print(get_entropy('Test', 9/14, 5/14, 1))
# print(get_entropy(class_label, pos, neg, tot))
# print(get_attribute_labels(df))


def print_tree():
    return ''

=======
    tree_level = 0
    left_node = 0
    right_node = 0
    while left_node is not None and right_node is not None:
        if tree_level == 0:
            if left_node is not None:
                left_node = dt.left
                print(dt.label, '=', left_node.label, ':')
            elif right_node is not None:
                right_node = dt.right
                print(dt.label, '=', right_node.label, ':')
        elif tree_level > 0:
            while left_node.left is not None:
                to_print = ''
                for i in range(tree_level):
                    to_print += '| '
                left_node = left_node.left
                left_branch_value = left_node.left
                left_branch_value = left_branch_value.label
                if type(left_branch_value) != str:
                    str(left_branch_value)
                to_print += left_node.label + ' = ' + left_branch_value + ' : '
                print(to_print)
                tree_level += 1
        tree_level += 1


>>>>>>> 674915b821c5632699b59476427ea1ad98a6fe85
# Computes the number of non-leaf
# nodes in a tree.
<<<<<<< HEAD
def count_non_leaf(root):
||||||| a3b21ab
def countNonleaf(root):

=======
def count_non_leaf(root):

>>>>>>> 674915b821c5632699b59476427ea1ad98a6fe85
    # Base cases.
    if root is None or (root.left is None and root.right is None):
        return 0

<<<<<<< HEAD
    # If root is Not None and its one of its child is also not None
    return 1 + count_non_leaf(root.left) + count_non_leaf(root.right)

||||||| a3b21ab
    # If root is Not None and its one of
    # its child is also not None
    return (1 + countNonleaf(root.left) +
                countNonleaf(root.right))
=======
    # If root is Not None and its one of
    # its child is also not None
    return 1 + count_non_leaf(root.left) + count_non_leaf(root.right)

>>>>>>> 674915b821c5632699b59476427ea1ad98a6fe85

# Implementation of the Post Pruning Algorithm
# Input: A decision tree DT, an integer L and an integer K
# Return: Post-pruned decision tree
<<<<<<< HEAD
# def post_pruning(l, k):
#     # Build a decision tree using all the training data
#     # This will most likely need to be done using ID3 which is on Evan's Branch
#     # Reminder to update this once we merge Evan's fork with our main codebase, commented for now
#     # decision_tree = id3(train_df, "Class", list(train_df.columns[0:-1]))
#     decision_tree = ""
#
#     # Optimal Decision Tree variable is initially set to the unpruned tree
#     optimal_d = decision_tree
#
#     # Begin pruning with a Loop from 1 to L
#     for x in [1, L]:
#         # Copy the current most optimal tree into a new tree d_prime
#         d_prime = optimal_d
#
#         # Select a random number M between 1 and K
#         # random.randrange(start, stop, step)
#         m = random.randrange(1, K, 1)
#
#         # Loop from 1 to m and start pruning subtrees
#         for y in range(1, m):
#             # Let n denote the number of non-leaf nodes in the decision tree d_prime
#             n = count_n_on_leaf(d_prime)
#
#             # Order the nodes in d_prime from 1 to n
#             # Reminder to implement
#             # node_list = ""
#
#             # Select a random number between 1 and n, assign it to p
#             p = random.randrange(1, n, 1)
#
#             # Replace the subtree rooted at P in d_prime with a leaf node.
#             # Assign the leaf node the value of the majority class
#             # Reminder to implement
#             """ Not sure exactly how to implement this. Need a better understanding of Evan's Tree Code """
#
#         # End inner loop and evaluate the accuracy of d_prime on the validation set
#         # First check the accuracy of our current optimal tree
#         # Reminder to implement this
#         optimal_d_accuracy = "Percent correctly classified examples"
#
#         # Calculate the accuracy of our newly pruned tree
#         d_prime_accuracy = "Percent correctly classified examples"
#
#         # If d_prime_accuracy > optimal_d_accuracy, replace optimal_d with d_prime
#         if d_prime_accuracy > optimal_d_accuracy:
#             optimal_d = d_prime
#
#     # Return the pruned tree with the best accuracy
#     return optimal_d
||||||| a3b21ab
def post_pruning(L, K):
    # Build a decision tree using all the training data
    #? This will most likely need to be done using ID3 which is on Evan's Branch
    #! Reminder to update this once we merge Evan's fork with our main codebase, commented for now
    #decision_tree = id3(train_df, "Class", list(train_df.columns[0:-1]))
    decision_tree = ""

    # Optimal Decision Tree variable is initially set to the unpruned tree
    optimal_d = decision_tree

    # Begin pruning with a Loop from 1 to L
    for x in [1, L]:
        # Copy the current most optimal tree into a new tree d_prime
        d_prime = optimal_d

        # Select a random number M between 1 and K
        # random.randrange(start, stop, step)
        m = random.randrange(1, K, 1)

        # Loop from 1 to m and start pruning subtrees
        for x in range(1, m):
            # Let n denote the number of non-leaf nodes in the decision tree d_prime
            n = countNonleaf(d_prime)

            # Order the nodes in d_prime from 1 to n
            #! Reminder to implement
            node_list = ""

            # Select a random number between 1 and n, assign it to p
            p = random.randrange(1, n, 1)

            # Replace the subtree rooted at P in d_prime with a leaf node.
            # Assign the leaf node the value of the majority class
            #! Reminder to implement
            """ Not sure exactly how to implement this. Need a better understanding of Evan's Tree Code """

        # End inner loop and evaluate the accuracy of d_prime on the validation set
        # First check the accuracy of our current optimal tree
        #! Reminder to implement this
        optimal_d_accuracy = "Percent correctly classified examples"

        # Calculate the accuracy of our newly pruned tree
        d_prime_accuracy = "Percent correctly classified examples"

        # If d_prime_accuracy > optimal_d_accuracy, replace optimal_d with d_prime
        if (d_prime_accuracy > optimal_d_accuracy):
            optimal_d = d_prime
=======
def post_pruning(dt, L, K):
    # Build a decision tree using all the training data
    #? This will most likely need to be done using ID3 which is on Evan's Branch
    #! Reminder to update this once we merge Evan's fork with our main codebase, commented for now

    # Optimal Decision Tree variable is initially set to the unpruned tree
    optimal_d = dt

    # Begin pruning with a Loop from 1 to L
    for x in [1, L]:
        # Copy the current most optimal tree into a new tree d_prime
        d_prime = optimal_d

        # Select a random number M between 1 and K
        # random.randrange(start, stop, step)
        m = random.randrange(1, K, 1)

        # Loop from 1 to m and start pruning subtrees
        for x in range(1, m):
            # Let n denote the number of non-leaf nodes in the decision tree d_prime
            n = count_non_leaf(d_prime)

            # Order the nodes in d_prime from 1 to n
            #! Reminder to implement
            node_list = ""

            # Select a random number between 1 and n, assign it to p
            p = random.randrange(1, n, 1)

            # Replace the subtree rooted at P in d_prime with a leaf node.
            # Assign the leaf node the value of the majority class
            #! Reminder to implement
            """ Not sure exactly how to implement this. Need a better understanding of Evan's Tree Code """

        # End inner loop and evaluate the accuracy of d_prime on the validation set
        # First check the accuracy of our current optimal tree
        #! Reminder to implement this
        optimal_d_accuracy = "Percent correctly classified examples"

        # Calculate the accuracy of our newly pruned tree
        d_prime_accuracy = "Percent correctly classified examples"

        # If d_prime_accuracy > optimal_d_accuracy, replace optimal_d with d_prime
        if d_prime_accuracy > optimal_d_accuracy:
            optimal_d = d_prime
>>>>>>> 674915b821c5632699b59476427ea1ad98a6fe85



# l = 1
# k = 0
# training_set = 'data_sets2/data_sets2/training_set.csv'
training_set = 'data_sets1/data_sets1/example_set.csv'
validation_set = 'data_sets2/data_sets2/validation_set.csv'
test_set = 'data_sets2/data_sets2/test_set.csv'
to_print = 'yes'

# run id3 algorithm

# convert training_set, validation_set, and test_set to dataframes
train_df = pd.read_csv(training_set)
validation_df = pd.read_csv(validation_set)
test_df = pd.read_csv(test_set)

# micro_set = 'data_sets2/data_sets2/micro_set.csv'
# micro_df = pd.read_csv(micro_set)

<<<<<<< HEAD
tree = id3(train_df, "PlayTennis", list(train_df.columns[0:-1]))
# tree = id3(micro_df, "Class", list(micro_df.columns[0:-1]))
standard_output_format(tree)
||||||| a3b21ab
tree = id3(train_df, "Class", list(train_df.columns[0:-1]))
#tree = id3(micro_df, "Class", list(micro_df.columns[0:-1]))
print(tree)
=======
micro2 = 'data_sets2/data_sets2/micro2.csv'
micro2_df = pd.read_csv(micro2)

# tree = id3(train_df, "Class", list(train_df.columns[0:-1]))
# tree = id3(micro_df, "Class", list(micro_df.columns[0:-1]))
tree = id3(micro2_df, "Class", list(micro2_df.columns[0:-1]))

printy(tree)
>>>>>>> 674915b821c5632699b59476427ea1ad98a6fe85
