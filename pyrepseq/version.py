from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = '2'  # use '' for first of series, number for 1 and above
#_version_extra = 'dev'
_version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 5 - Production/Stable",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python :: 3",              
               "Programming Language :: Python :: 3.8",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "Python library for immune repertoire analyses" 
# Long description will go up on the pypi page
long_description = """
Pyrepseq is a library for facilitating high-level immune repertoire analyses.
"""

NAME = "pyrepseq"
MAINTAINER = "Andreas Mayer"
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
PACKAGES = ['pyrepseq']
PACKAGE_DATA = {}
REQUIRES = ["numpy", "scipy", "pandas", "Levenshtein", "matplotlib",
            "seaborn", "logomaker", "biopython"]
