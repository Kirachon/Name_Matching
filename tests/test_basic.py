def test_basic():
    """Basic test function"""
    assert True


def test_package_imports():
    """Test that all package imports work correctly."""
    import src
    from src import NameMatcher
    from src import parse_name, standardize_name, jaro_winkler_similarity

    assert src.__version__ == "0.1.0"
    assert callable(parse_name)
    assert callable(standardize_name)
    assert callable(jaro_winkler_similarity)
    assert isinstance(NameMatcher(), NameMatcher)
