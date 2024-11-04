/// Returns the index of the sibling, given an index.
#[inline]
pub(crate) fn sibling_index(index: usize) -> Option<usize> {
    if index == 0 {
        None
    } else if is_left_child(index) {
        Some(index + 1)
    } else {
        Some(index - 1)
    }
}

/// Returns the index of the parent, given an index.
#[inline]
pub(crate) fn parent_index(index: usize) -> Option<usize> {
    if index > 0 {
        Some((index - 1) >> 1)
    } else {
        None
    }
}

/// Returns the index of the left child, given an index.
#[inline]
pub(crate) fn left_child_index(index: usize) -> usize {
    2 * index + 1
}

/// Returns the index of the right child, given an index.
#[inline]
pub(crate) fn right_child_index(index: usize) -> usize {
    2 * index + 2
}

#[inline]
pub(crate) fn convert_index_to_last_level(
    index: usize,
    tree_height: usize,
) -> usize {
    index + (1 << (tree_height - 1)) - 1
}

/// Returns true iff the given index represents a left child.
#[inline]
pub(crate) fn is_left_child(index: usize) -> bool {
    index % 2 == 1
}
