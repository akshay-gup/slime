import warnings

from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions


def test_get_seqlen_balanced_partitions_equal_size_divisible():
    seqlen_list = [2, 3, 5, 7]

    partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=2, equal_size=True)

    assert len(partitions) == 2
    assert sorted(len(partition) for partition in partitions) == [2, 2]
    assert sorted(idx for partition in partitions for idx in partition) == [0, 1, 2, 3]


def test_get_seqlen_balanced_partitions_equal_size_not_divisible_falls_back():
    seqlen_list = [2, 3, 5, 7, 11]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=2, equal_size=True)

    assert len(caught) == 1
    assert "Falling back to variable-size partitions" in str(caught[0].message)
    assert len(partitions) == 2
    assert sorted(idx for partition in partitions for idx in partition) == [0, 1, 2, 3, 4]
    assert sorted(len(partition) for partition in partitions) == [2, 3]
