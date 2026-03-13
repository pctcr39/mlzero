"""
setup.py — Package installation config.

Run once to install mlzero as an editable package:
    pip install -e .

After this, you can import from anywhere:
    from mlzero.supervised.regression.linear import LinearRegression
    from mlzero.core.losses import mse
"""

from setuptools import setup, find_packages

setup(
    name="mlzero",
    version="0.1.0",
    description="Machine Learning from Zero — educational ML library",
    package_dir={"": "src"},        # tell Python: packages live in src/
    packages=find_packages(where="src"),  # auto-find all packages in src/
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "matplotlib>=3.7",
        "scikit-learn>=1.3",
        "pandas>=2.0",
    ],
    extras_require={
        "deep_learning": ["torch>=2.0"],
        "notebooks":     ["jupyter>=1.0", "ipykernel>=6.0"],
        "dev":           ["pytest>=7.0", "pytest-cov>=4.0"],
    },
)
