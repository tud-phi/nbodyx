import pytest


def add(a, b):
    return a + b


def test_add():
    assert add(2, 3) == 5


def test_int_hello():
    """
    This test is marked implicitly as an integration test because the name contains "_init_"
    https://docs.pytest.org/en/6.2.x/example/markers.html#automatically-adding-markers-based-on-test-names
    """
    test_add()
