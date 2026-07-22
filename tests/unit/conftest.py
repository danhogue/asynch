import pytest


@pytest.fixture(scope="session")
def initialize_tests():
    yield


@pytest.fixture
def truncate_table():
    yield
