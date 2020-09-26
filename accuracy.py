from tree_traversal import tree_traversal

def measure_accuracy(data, dt):
    """This function is used to measure the accuracy of a trained decision tree on new data (such as a test or
    validation set.

    Arguments:
    data    a pandas dataframe
    dt      a decision tree Node
    """
    x = data.iloc[:, :-1]
    count = 0

    # i is the index of the row, "row" is the data in the row.
    # For each row, make a dict of column names and values to pass to
    # tree_traversal().
    for i, row in x.iterrows():
        check_row = {}
        for j in range(len(x.columns)):
            check_row[x.columns[j]] = row[j]
        traversal_result = tree_traversal(dt, check_row)
        dataset_result = data['Class'][i]
        if traversal_result is not None:
            if (int(traversal_result) == dataset_result):
                count += 1
    percent_accuracy = count/len(data) * 100
    return percent_accuracy
