# Module: test_package.py
# Location: tests/test_package.py
# Purpose: Verify that the rag_performance package can be imported successfully.
# Description: Provides a minimal foundation-level test for the project package.

# Import the package's file location under a non-conflicting name.
from rag_performance import __file__ as package_file


# Test that the rag_performance package can be imported successfully.
def test_package_is_importable() -> None:
    assert package_file
