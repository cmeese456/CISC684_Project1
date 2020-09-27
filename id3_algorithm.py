import math
import copy
import random
import sys
from tree import *

"""This file is for implementing the ID3 decision tree learning algorithm. As discussed in class,
the main step in decision tree learning is choosing the next attribute to split on. Implement the
following two heuristics for selecting the next attribute.

1. Information Gain heuristic (see Mitchell slides Ch. 3)

2. Variance impurity heuristic (described below)"""

def print_output(node, depth=0):
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
    prefix = "| " * depth
    if node:
        if node.left.left.left or node.left.left.right:
            print(prefix + node.label + " = " + str(node.left.label) + " : ")
            print_output(node.left.left, depth + 1)
        else:
            print(prefix + node.label + " = " + str(node.left.label) + " : " + str(node.left.left.label))
        if node.right.left.left or node.right.left.right:
            print(prefix + node.label + " = " + str(node.right.label) + " : ")
            print_output(node.right.left, depth + 1)
        else:
            print(prefix + node.label + " = " + str(node.right.label) + " : " + str(node.right.left.label))

# Get the label (i.e. attribute name) with the highest information gain.
def information_gain_heuristic(s, attr_list, df):
    gain_list = []
    for attr in attr_list:
        gain_list.append(information_gain(s, attr, df))
    return max(gain_list, key=lambda item: item[1])[0]


# Get the information gain for an attribute.
def information_gain(s, attr, df):
    s_pos = len(df.loc[(df[s] == 1)])  # Number of positive (i.e. Class=1) examples
    s_neg = len(df.loc[(df[s] == 0)])  # Number of negative (i.e. Class=0) examples
    s_total = len(df.index)  # Total number of examples
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
            print(df.iloc[i])
            print(df)
            sys.exit()
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
    gain = entropy_s[1] - (attr_val0_total / attr_total) * entropy_attr_0[1] - (attr_val1_total / attr_total) * \
           entropy_attr_1[1]
    # Return the label of the attribute and its gain
    return attr, gain


def get_entropy(node_label, val1_instances, val0_instances, total):
    """The formula for entropy is Entropy = -p_1 * log_2(p_1) - p_0 * log_2(p_0)"""
    """As per Mitchell p. 56, 0 log 0 is defined as zero """
    # If there are no more examples with this value for this attribute, Return
    # entropy = 0. This avoids a divide by zero error.
    if total == 0:
        entropy = 0
    else:
        entropy = (-1 * val1_instances / total * math.log2(
            val1_instances / total if val1_instances / total > 0 else 1)) - \
                  (val0_instances / total * math.log2(val0_instances / total if val0_instances / total > 0 else 1))
    return node_label, entropy


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
        for i in [0, 1]:
            # The code below creates a node in-between a parent node and its
            # child. The sole purpose of this in-between node is to bear a label
            # reflecting whether the parent node's value was 0 or 1 for the
            # child branch. BUT that information is already conveyed by the
            # order in which nodes are inserted. The first insertion goes to the
            # left and signifies a "0" value for the parent node. The second
            # goes on the right and signifies a "1" value for the parent node.
            # So this in-between node is redundant.
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
    return root


# This function performs the variance impurity calculation across all attributes.
def variance_impurity_heuristic(s, attr_list, df):
    # Keep a list of the variance gain of each attribute
    gain_list = []
    # Calculate the variance impurity gain of each attribute in the list
    for attr in attr_list:
        gain_list.append(variance_impurity_gain(s, attr, df))
    # Return the label of the attribute with the highest gain from the list
    return max(gain_list, key=lambda item: item[1])[0]


def variance_impurity_gain(s, attr, df):
    s_pos = len(df.loc[(df[s] == 1)])  # Number of positive (i.e. Class=1) examples
    s_neg = len(df.loc[(df[s] == 0)])  # Number of negative (i.e. Class=0) examples
    s_total = len(df.index)  # Total number of examples
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
            print(df.iloc[i])
            print(df)
            sys.exit()
    attr_val0_total = len(df.loc[(df[attr] == 0)])
    attr_val1_total = len(df.loc[(df[attr] == 1)])
    attr_total = len(df.index)
    # Calculate the Variance Impurity of the entire set
    variance_s = variance_impurity(s, s_pos, s_neg, s_total)
    # Calculate the Variance Impurtiy of the subset that has "0" as the attribute value
    try:
        variance_attr_0 = variance_impurity(attr, attr_val0_pos, attr_val0_neg, attr_val0_total)
    except ZeroDivisionError:
        print(attr)
        print(df)
        sys.exit()
    # Calculate the Variance Impurity of the subset that has "1" as the attribute value
    try:
        variance_attr_1 = variance_impurity(attr, attr_val1_pos, attr_val1_neg, attr_val1_total)
    except ZeroDivisionError:
        print(attr)
        print(df)
        sys.exit()
    # Subtract from the set's variance the variance of each value multiplied by its proportion in the set
    gain = variance_s[1] - ((attr_val0_total / attr_total) * variance_attr_0[1]) - ((attr_val1_total / attr_total) * \
        variance_attr_1[1])
    # Return the label of the attribute and its gain
    return attr, gain

# Variance Impurity = (K0/K)*(K1/K)
def variance_impurity(node_label, val1_instances, val0_instances, total):
    if total == 0:
        variance_impurity = 0
    else:
        # Calculate the variance impurity using the formula from the homework
        variance_impurity = (val0_instances / total) * (val1_instances / total)
    # Return the impurity and the label
    return node_label, variance_impurity

# examples_list     A pandas dataframe
# target_attribute  The name of the target attribute (i.e. "Class")
# attributes_list   A list of attribute names (i.e. "XC", "XD", etc.)
def id3_variance_impurity(examples_list, target_attribute, attributes_list):
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
        A = variance_impurity_heuristic(target_attribute, attributes_list, examples_list)
        root.label = A
        for i in [0, 1]:
            # The code below creates a node in-between a parent node and its
            # child. The sole purpose of this in-between node is to bear a label
            # reflecting whether the parent node's value was 0 or 1 for the
            # child branch. BUT that information is already conveyed by the
            # order in which nodes are inserted. The first insertion goes to the
            # left and signifies a "0" value for the parent node. The second
            # goes on the right and signifies a "1" value for the parent node.
            # So this in-between node is redundant.
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
                new_branch.insert(id3_variance_impurity(examples_list_vi, target_attribute, trimmed_attributes))
                root.insert(new_branch)
    return root
