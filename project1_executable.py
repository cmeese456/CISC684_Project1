import sys
import pandas as pd
from id3_algorithm import *
from accuracy import measure_accuracy
from post_prune import *

# get command line arguments ************ UNCOMMENT AFTER FINISHED DEVELOPING *********
# l = sys.argv[0]
# k = sys.argv[1]
# training_set = sys.argv[2]
# validation_set = sys.argv[3]
# test_set = sys.argv[4]
# to_print = sys.argv[5]

# if len(sys.argv) != 6:
#     print('Incorrect Usage! Please enter six arguments: <L>, <K>, <training-set>, <validation-set>, <test-set>, '
#           '<to-print>')
#     sys.exit(2)

l = 4
k = 3
training_set = 'data_sets2/data_sets2/training_set.csv'
validation_set = 'data_sets2/data_sets2/validation_set.csv'
test_set = 'data_sets2/data_sets2/test_set.csv'
to_print = 'yes'

# check that l and k are integers, if not exit the program
if type(l) is not int:
    sys.stderr.write('L is not an integer. Please restart the program with L as an integer')
    sys.exit()

if type(k) is not int:
    sys.stderr.write('K is not an integer. Please restart the program with K as an integer')
    sys.exit()

# convert training_set, validation_set, and test_set to dataframes
train_df = pd.read_csv(training_set)
validation_df = pd.read_csv(validation_set)
test_df = pd.read_csv(test_set)

# create a tree by first performing the ID3 algorithm
# this tree is built using ID3 with Information Gain Heuristic

tree_information_gain = id3(train_df, 'Class', list(train_df.columns[0:-1]))
vi_tree = id3_variance_impurity(train_df, 'Class', list(train_df.columns[0:-1]))

# Create a second tree by first performing the ID3 algorithm
# this tree is built using ID3 with the Variance Impurity heuristic
#vi_tree = id3_variance_impurity(train_df, 'Class', list(train_df.columns[0:-1]))

# perform post-pruning on the tree
prune_info_gain_tree = post_pruning(l, k, tree_information_gain, validation_df)

# create a tree for variance impurity by first performing the ID3 algorithm
# vi_tree = id3(train_df, 'Class', list(train_df.columns[0:-1]))

# perform post-pruning on the variance impurity tree
prune_vi_tree = post_pruning(l, k, vi_tree, validation_df)

if to_print == 'yes':
    print('Printing Info Gain Decision Tree')
    printy(tree_information_gain)
    print('Accuracy of Info Gain Decision Tree pre-pruning: ' + str(measure_accuracy(test_df, tree_information_gain)) + '%')
    print('Printing Variance Impurity Tree pre-pruning')
    printy(vi_tree)
    print('Accuracy of Variance Impurity Decision Tree pre-pruning: ' + str(measure_accuracy(test_df, vi_tree)) + '%')
    print('Accuracy of Information Gain Tree post-pruning:', measure_accuracy(test_df, prune_info_gain_tree))
    print('Accuracy of Variance Impurity Tree post-pruning:', measure_accuracy(test_df, prune_vi_tree))

'''
# Debugging tests
micro_set = 'data_sets2/data_sets2/micro3.csv'
micro_df = pd.read_csv(micro_set)
micro_valid = 'data_sets2/data_sets2/micro3valid.csv'
microvalid_df = pd.read_csv(micro_valid)
micro_test = 'data_sets2/data_sets2/micro3test.csv'
microtest_df = pd.read_csv(micro_test)
tree_information_gain = id3(micro_df, 'Class', list(micro_df.columns[0:-1]))
vi_tree = id3_variance_impurity(micro_df, 'Class', list(micro_df.columns[0:-1]))
prune_info_gain_tree = post_pruning(l, k, tree_information_gain, microvalid_df)
prune_vi_tree = post_pruning(l, k, vi_tree, microvalid_df)
print('Printing Info Gain Decision Tree')
#printy(tree_information_gain)
print('Accuracy of Info Gain Decision Tree pre-pruning: ' + str(measure_accuracy(microtest_df, tree_information_gain)) + '%')
print('Printing Variance Impurity Tree pre-pruning')
#printy(vi_tree)
print('Accuracy of Variance Impurity Decision Tree pre-pruning: ' + str(measure_accuracy(microtest_df, vi_tree)) + '%')
print('Printing Variance Impurity Tree post-pruning')
print('Accuracy of Information Gain Tree post-pruning:', measure_accuracy(microtest_df, prune_info_gain_tree))
print('Accuracy of Variance Impurity Tree post-pruning:', measure_accuracy(microtest_df, prune_vi_tree))
'''
