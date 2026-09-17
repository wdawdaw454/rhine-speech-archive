from src.evaluation.cer import cer, char_edit_distance


def test_cer_perfect_match():
    assert cer("我们明天开会", "我们明天开会") == 0.0


def test_cer_single_substitution():
    assert cer("我们明天开会", "我们后天开会") == 1 / 6


def test_cer_insertion_deletion():
    assert cer("abc", "a bc") == 1 / 3


def test_cer_empty_reference():
    assert cer("", "abc") == 1.0


def test_cer_empty_hypothesis():
    assert cer("abc", "") == 1.0


def test_edit_distance_counts():
    assert char_edit_distance("abc", "abc") == 0
    assert char_edit_distance("abc", "axc") == 1
    assert char_edit_distance("", "abc") == 3