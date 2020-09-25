def tree_traversal(dt):
    traversal_return = ''
    if dt:
        if dt.left.left.left or dt.left.left.right:
            traversal_return = dt.left.label
            tree_traversal(dt.left.left)
        elif dt.right.left.left or dt.right.left.right:
            traversal_return = dt.right.label
            tree_traversal(dt.right.left)
    return traversal_return
