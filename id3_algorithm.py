import pandas as pd
import sys
import math

"""This file is for implementing the ID3 decision tree learning algorithm. As discussed in class,
the main step in decision tree learning is choosing the next attribute to split on. Implement the
following two heuristics for selecting the next attribute.

1. Information Gain heuristic (see Mitchell slides Ch. 3)

2. Variance impurity heuristic (described below)"""


df = pd.read_csv('data_sets2/data_sets2/training_set.csv')


def get_entropy(node_label, positive_instances, negative_instances, total):
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



def information_gain_heuristic():
    return ''

#! Note: Variance Impurity Heuristics are based off Evan's work on the Node and Attribute Classes
#! They will only work once we merge this code with his side branch
# This function calculates the variance impurity of all attributes in attr_dict in comparison to the total set s
#! Important: attr_dict must only contain attributes to be tested, prune the input based on tree level before calling
def variance_impurity_heuristic(s, attr_dict):
    # Keep a list of the variance gain of each attribute
    gain_list = []

    # Calculate the variance impurity gain of each attribute in the list
    for attr in attr_dict:
        gain_list.append(variance_impurity_gain(s, attr_dict[attr]))

    # Return the label of the attribute with the highest gain from the list
    return max(gain_list, key=lambda item:item[1])[0]

def variance_impurity_gain(s, attr):
    # Get the variance impurity of the entire set
    variance_s = variance_impurity(s.label, s.getTotalVal1(), s.getTotalVal0(), s.getTotal())

    # Get the variance impurity of the subset that has "0" as a value for the attribute
    variance_attr_0 = variance_impurity(attr.label, attr.val0_pos, attr.val0_neg, attr.getTotalVal0())

    # Get the variance impurity of the subset that has "1" as the value for the attribute
    variance_attr_1 = variance_impurity(attr.label, attr.val1_pos, attr.val1_neg, attr.getTotalVal0())

    # Subtract the weighted variance impurity of each value from that of the total set
    variance_gain = variance_s[1] - ((attr.getTotalVal0() / attr.getTotal()) * variance_attr_0[1] - (attr.getTotalVal1() / attr.getTotal()) * variance_attr_1[1])

    # Return label and gain
    return attr.label, variance_gain

# Variance Impurity = (K0/K)*(K1/K)
def variance_impurity(node_label, val1_instances, val0_instances, total):
    # Calculate the variance impurity using the formula from the homework
    variance_impurity = (val1_instances / total) * (val0_instances / total)

    # Return the impurity and the label
    return node_label, variance_impurity



def standard_output_format():
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
    return ''


# class_label, pos, neg, tot = get_class_labels(df)
# print(get_entropy('Test', 9/14, 5/14, 1))
# print(get_entropy(class_label, pos, neg, tot))
# print(get_attribute_labels(df))


def print_tree():
    return ''
