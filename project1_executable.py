import sys
import pandas as pd
from id3_algorithm import *
from accuracy import measure_accuracy
from post_prune import *

# get command line arguments
# check that l and k are integers, if not exit the program
try:
    l = int(sys.argv[1])
except ValueError:
    sys.stderr.write(str(sys.argv[1]) + ' is not an integer. Please restart the program with arg L as an integer\n')
    sys.exit()
try:
    k = int(sys.argv[2])
except ValueError:
    sys.stderr.write(str(sys.argv[2]) + ' is not an integer. Please restart the program with K as an integer\n')
    sys.exit()
training_set = sys.argv[3]
validation_set = sys.argv[4]
test_set = sys.argv[5]
to_print = sys.argv[6]

if len(sys.argv) != 7:
    print('Incorrect Usage! Please enter six arguments: <L>, <K>, <training-set>, <validation-set>, <test-set>, <to-print>')
    sys.exit(2)

# convert training_set, validation_set, and test_set to dataframes
try:
    train_df = pd.read_csv(training_set)
except FileNotFoundError:
    sys.stderr.write('Error: File ' + training_set + ' not found.\n')
try:
    validation_df = pd.read_csv(validation_set)
except FileNotFoundError:
    sys.stderr.write('Error: File ' + validation_set + ' not found.\n')
try:
    test_df = pd.read_csv(test_set)
except FileNotFoundError:
    sys.stderr.write('Error: File ' + test_set + ' not found.\n')

# Create trees using ID3 with information gain and variance impurity heuristics.
info_gain_tree = id3(train_df, 'Class', list(train_df.columns[0:-1]))
vi_tree = id3_variance_impurity(train_df, 'Class', list(train_df.columns[0:-1]))

# Perform post-pruning on the information gain and variance impurity trees.
prune_info_gain_tree = post_pruning(l, k, info_gain_tree, validation_df)
prune_vi_tree = post_pruning(l, k, vi_tree, validation_df)

info_gain_preprune_accur = measure_accuracy(test_df, info_gain_tree)
vi_preprune_accur = measure_accuracy(test_df, vi_tree)
info_gain_postprune_accur = measure_accuracy(test_df, prune_info_gain_tree)
vi_postprune_accur = measure_accuracy(test_df, prune_vi_tree)

if to_print == 'yes':
    print('Printing Variance Impurity Tree post-pruning')
    print_output(prune_vi_tree)
    print('Printing Information Gain Tree post-pruning')
    print_output(prune_info_gain_tree)
    print()
    print('{0:8s}      {1:20s}        {2:20s}'.format('Params', 'Pre-Prune Accuracy', 'Post-Prune Accuracy'))
    print('{0:4s}  {1:4s}    {2:10s}  {3:10s}      {2:10s}  {3:10s}'.format('L', 'K', 'VarImp', 'InfoGain'))

print('{0:<4d}  {1:<4d}    {2:<9.4f}%  {3:<9.4f}%      {4:<9.4f}%  {5:<9.4f}%'.format(l, k, vi_preprune_accur,
                                                                                      info_gain_preprune_accur,
                                                                                      vi_postprune_accur,
                                                                                      info_gain_postprune_accur))
