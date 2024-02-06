from setuptools import find_packages

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 1
_version_minor = 3
_version_micro = ""  # use '' for first of series, number for 1 and above
# _version_extra = 'dev'
_version_extra = ""  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering",
]

# Description should be a one-liner:
description = "Python library for immune repertoire analyses"
# Long description will go up on the pypi page
long_description = """
Pyrepseq = scipy & seaborn for studying adaptive immunity: modular implementations of algorithms for fast analyses, and bespoke plotting functions for compelling visualizations.


## Documentation and examples

You can find API documentation on [readthedocs](https://pyrepseq.readthedocs.io/en/latest/?badge=latest)

There are jupyter notebooks illustrating some of the functionality in the 'examples' folder.

You can also find usage examples by looking at the code underlying our recent paper [Mayer Callan PNAS 2023](https://github.com/andim/paper_coincidences).

## Installation

The quickest way to install Pyrepseq is via pip:

`pip install pyrepseq`


Pyrepseq can also be installed from its [Github](https://github.com/andim/pyrepseq) source, by running `python setup.py install` in the main directory.

## Support and contributing

For bug reports and enhancement requests use the [Github issue tool](http://github.com/andim/pyrepseq/issues/new), or (even better!) open a [pull request](http://github.com/andim/pyrepseq/pulls) with relevant changes. If you have any questions don't hesitate to contact us by email (andimscience@gmail.com) or Twitter ([@andimscience](http://twitter.com/andimscience)).

You can run the testsuite by running `pytest` in the top-level directory.

You are cordially invited to [contribute](https://github.com/andim/pyrepseq/blob/master/CONTRIBUTING.md) to the further development of pyrepseq!
"""

NAME = "pyrepseq"
MAINTAINER = "Andreas Tiffeau-Mayer"
MAINTAINER_EMAIL = "andimscience@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://pyrepseq.readthedocs.io/"
DOWNLOAD_URL = "http://github.com/andim/pyrepseq"
LICENSE = "MIT"
AUTHOR = "Andreas Mayer"
AUTHOR_EMAIL = "andimscience@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGES = find_packages()
PACKAGE_DATA = {"": ["data/*.csv"]}
REQUIRES = [
    "numpy",
    "scipy",
    "pandas",
    "rapidfuzz",
    "Levenshtein",
    "matplotlib",
    "seaborn",
    "logomaker",
    "biopython",
    "tcrdist3",
    "tidytcells~=2.0",
]
