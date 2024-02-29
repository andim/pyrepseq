import os

from setuptools import find_packages, setup

ver_file = os.path.join("pyrepseq", "version.py")
with open(ver_file) as f:
    exec(f.read())

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
Pyrepseq = scipy & seaborn for studying adaptive immunity: modular algorithms for fast analyses, and bespoke plotting functions for compelling visualizations.


## Documentation and examples

You can find API documentation on [readthedocs](https://pyrepseq.readthedocs.io/en/latest/?badge=latest)

There are also Jupyter notebooks illustrating some of the functionality in the 'examples' [folder](https://github.com/andim/pyrepseq/tree/main/examples).

You can also find usage examples in the repository accompanying our recent paper [Mayer Callan PNAS 2023](https://github.com/andim/paper_coincidences).

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
AUTHOR = "Q-Immuno Lab (PI: Andreas Tiffeau-Mayer)"
AUTHOR_EMAIL = "qimmuno@gmail.com"
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
    "igraph",
    "scikit-learn",
    "logomaker",
    "biopython",
    "tcrdist3",
    "tidytcells~=2.0",
]

opts = dict(
    name=NAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    platforms=PLATFORMS,
    version=VERSION,
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    include_package_data=True,
    install_requires=REQUIRES,
)


if __name__ == "__main__":
    setup(**opts)
