def tree_traversal(dt, tree_row):
    traversal_return = ''
    if dt.left is None and dt.right is None:
        traversal_return = dt.label
    elif dt.left is not None:
        traversal_return = tree_traversal(dt.left)
    elif dt.right is not None:
        traversal_return = tree_traversal(dt.right)
    return traversal_return
