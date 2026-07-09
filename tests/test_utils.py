"""Tests for form4lab/utils.py shared utilities."""

from form4lab.utils import to_python_float, is_csuite


# --- to_python_float ---

def test_to_python_float_none():
    assert to_python_float(None) is None


def test_to_python_float_int():
    assert to_python_float(42) == 42.0
    assert isinstance(to_python_float(42), float)


# --- is_csuite ---

def test_is_csuite_ceo():
    assert is_csuite("CEO") is True


def test_is_csuite_cfo():
    assert is_csuite("CFO") is True


def test_is_csuite_president():
    assert is_csuite("President") is True


def test_is_csuite_chairman():
    assert is_csuite("Chairman of the Board") is True


def test_is_csuite_chief_executive_officer():
    assert is_csuite("Chief Executive Officer") is True


def test_is_csuite_vice_president_excluded():
    assert is_csuite("Vice President") is False


def test_is_csuite_evp_excluded():
    assert is_csuite("Executive Vice President") is False


def test_is_csuite_director_excluded():
    assert is_csuite("Director") is False


def test_is_csuite_10pct_owner_excluded():
    assert is_csuite("10% Owner") is False


def test_is_csuite_none():
    assert is_csuite(None) is False


def test_is_csuite_empty_string():
    assert is_csuite("") is False


def test_is_csuite_coo():
    assert is_csuite("COO") is True


def test_is_csuite_cto():
    assert is_csuite("CTO") is True
