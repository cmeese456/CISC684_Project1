import sys
import pandas as pd
from CISC684_Project1.id3_algorithm import *

# get command line arguments ************ UNCOMMENT AFTER FINISHED DEVELOPING *********
# l = sys.argv[0]
# k = sys.argv[1]
# training_set = sys.argv[2]
# validation_set = sys.argv[3]
# test_set = sys.argv[4]
# to_print = sys.argv[5]

# l = 1
# k = 0
training_set = 'data_sets2/data_sets2/training_set.csv'
validation_set = 'data_sets2/data_sets2/validation_set.csv'
test_set = 'data_sets2/data_sets2/test_set.csv'
to_print = 'yes'

# run id3 algorithm

# convert training_set, validation_set, and test_set to dataframes
train_df = pd.read_csv(training_set)
validation_df = pd.read_csv(validation_set)
test_df = pd.read_csv(test_set)

# perform ID3 algorithm
# First, get the attribute label, total number of 1s, 0s, and the total number of values for each column
attribute_labels, attribute_positives, attribute_negatives, attribute_totals = get_attribute_labels(train_df)

# Next, get the output variable's (Class variable) data
class_label = attribute_labels[len(attribute_labels) - 1]
positive_class_instances = attribute_positives[len(attribute_positives) - 1]
negative_class_instances = attribute_negatives[len(attribute_negatives) - 1]
total = attribute_totals[len(attribute_totals) - 1]

# For every attribute:
#   a.) Calculate entropy
entropy_of_attributes = []
for i in range(len(attribute_labels)):
    entropy_of_attributes.append(get_entropy(attribute_labels[i], attribute_positives[i],
                                             attribute_negatives[i], attribute_totals[i]))

# Get the entropy for the Class
class_entropy = entropy_of_attributes[len(entropy_of_attributes) - 1]

#   b.) Take average information entropy for the current attribute
#   c.) Calculate gain for the current attribute


# if to_print == 'yes':
#     print_tree()
