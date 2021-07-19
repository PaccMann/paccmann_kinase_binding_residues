"""Package installer."""
import re
import os
import io
from setuptools import setup
from setuptools import find_packages

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open("pkbr/__init__.py", encoding="utf8").read(),
).group(1)

LONG_DESCRIPTION = ""
if os.path.exists("README.md"):
    with open("README.md") as fp:
        LONG_DESCRIPTION = fp.read()

scripts = []

setup(
    name="pkbr",
    version="0.0.1",
    description="PaccMann on kinase binding residues",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="PaccMann team",
    author_email=("tte@zurich.ibm.com, jab@zurich.ibm.com"),
    url="https://github.com/PaccMann/paccmann_kinase_binding_residues",
    license="MIT",
    install_requires=[
        "loguru",
        "numpy",
        "pandas",
        "paccmann_chemistry @ git+https://git@github.com/PaccMann/paccmann_chemistry/@0.0.4",
        "paccmann_gp @ git+https://git@github.com/PaccMann/paccmann_gp/@0.1.0",
        "paccmann_predictor @ git+https://git@github.com/PaccMann/paccmann_predictor/@0.0.4",
        "python-Levenshtein",
        "pytoda @ git+https://git@github.com/PaccMann/paccmann_datasets/@0.2.5",
        "pytorch_lightning",
        "rdkit-pypi",
        "torch",
        "tqdm",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages("."),
    scripts=scripts,
)
