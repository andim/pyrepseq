[![License](https://img.shields.io/pypi/l/pyrepseq.svg)](https://github.com/andim/pyrepseq/blob/master/LICENSE)
[![Latest release](https://img.shields.io/pypi/v/pyrepseq.svg)](https://pypi.python.org/pypi/pyrepseq)
[![Build Status](https://app.travis-ci.com/andim/pyrepseq.svg?branch=main)](https://app.travis-ci.com/andim/pyrepseq)
[![Documentation Status](https://readthedocs.org/projects/pyrepseq/badge/?version=latest)](https://pyrepseq.readthedocs.io/en/latest/?badge=latest)

# Pyrepseq: the immune repertoire analysis toolkit

Pyrepseq is `scipy` & `seaborn` for studying adaptive immunity: modular implementations of algorithms for fast analyses, and bespoke plotting functions for compelling visualizations.

## Documentation and examples

You can find API documentation on [readthedocs](https://pyrepseq.readthedocs.io/en/latest/?badge=latest).
You can also create a local copy of the API documentation by running:

```bash
make html
```

in the docs folder.

There are jupyter notebooks illustrating some of the functionality in the 'examples' folder.
You can also find usage examples by looking at the code underlying our recent paper [Mayer Callan PNAS 2023](https://github.com/andim/paper_coincidences).

## Installation

The quickest way to install Pyrepseq is via pip:

```bash
pip install pyrepseq[full]
```

This will install pyrepseq with all optional dependencies. Depending on whether dependencies are already installed this might take a few minutes.
You can also install the leading edge development version using:

```bash
pip install git+https://github.com/andim/pyrepseq
```

As the TCRdist dependency on parasail is known to cause installation issues on Mac OSX, pyrepseq can also be installed without this dependency by running:

```bash
pip install pyrepseq
```

In this case some functionality will not be available.
To allow installation to proceed on mac you might have to manually install build tools using:

```bash
brew install autoconf automake libtool
```

Pyrepseq can also be installed from its [Github](https://github.com/andim/pyrepseq) source, by running:

```bash
python setup.py install
```

in the main directory.

## Support and contributing

For bug reports and enhancement requests use the [Github issue tool](http://github.com/andim/pyrepseq/issues/new), or (even better!) open a [pull request](http://github.com/andim/pyrepseq/pulls) with relevant changes.
If you have any questions don't hesitate to contact us by email (andimscience@gmail.com) or Twitter ([@andimscience](http://twitter.com/andimscience)).

You can run the testsuite by running `pytest` in the top-level directory.
Dependencies for generating testing and generating local versions of the documentation can be installed using:

```bash
pip install pyrepseq[dev]
```

You are cordially invited to [contribute](https://github.com/andim/pyrepseq/blob/master/CONTRIBUTING.md) to the further development of pyrepseq!
