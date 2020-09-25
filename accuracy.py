from tree_traversal import tree_traversal


def measure_accuracy(data, dt):
    """This function is used to measure the accuracy of a trained decision tree on new data (such as a test or
    validation set."""
    x = data.iloc[:, :-1]
    count = 0
    result_in_tree = []

    for i, row in x.iterrows():
        traversal_result = tree_traversal(dt)
        result_in_tree.append(traversal_result)
        dataset_result = data['Class'][i]
        if int(traversal_result) == dataset_result:
            count += 1

    # calculate tree accuracy as a percentage
    percent_accuracy = count/len(data) * 100
    return percent_accuracy
