def tree_traversal(dt, row):
    '''
    Follow a row of a test or validation set through a decision tree and
    return a leaf.

    Arguments:
    dt      a decision tree Node
    row     a dict mapping column names to values for a given row in a dataframe
    '''
    traversal_return = None
    if dt.left or dt.right:
        follow_attribute = dt.label
        if int(row[follow_attribute]) == 0:
            traversal_return = tree_traversal(dt.left.left, row)
        elif int(row[follow_attribute]) == 1:
            traversal_return = tree_traversal(dt.right.left, row)
        else:
            sys.stderr.write('Illegal value found in leaf node.\n')
            sys.exit()
    else:
        traversal_return = dt.label
    return traversal_return
