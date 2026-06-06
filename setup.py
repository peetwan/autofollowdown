"""Compatibility shim. All real metadata lives in pyproject.toml.

This lets older pip/setuptools versions install the package (including editable
`pip install -e .`) even though the project is configured via pyproject.toml.
"""

from setuptools import setup

setup()
