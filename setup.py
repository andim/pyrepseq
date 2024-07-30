import os
from pathlib import Path
from setuptools import find_packages, setup

PROJECT_ROOT = Path(__file__).parent.resolve()


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

NAME = "pyrepseq"
MAINTAINER = "Andreas Tiffeau-Mayer"
MAINTAINER_EMAIL = "andimscience@gmail.com"
DESCRIPTION = "Python library for immune repertoire analyses"
LONG_DESCRIPTION = (PROJECT_ROOT/"README.md").read_text(encoding="utf-8")
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "http://pyrepseq.readthedocs.io/"
DOWNLOAD_URL = "http://github.com/andim/pyrepseq"
LICENSE = "MIT"
AUTHOR = "Q-Immuno Lab (PI: Andreas Tiffeau-Mayer)"
AUTHOR_EMAIL = "qimmuno@gmail.com"
PLATFORMS = "OS Independent"
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
    "tidytcells~=2.0",
    "tqdm",
]

DEV_DEPENDENCIES = [
    "pytest",
    "pytest-cov",
    "sphinx",
    "sphinx-rtd-theme",
]

# the parasail dependency of TCRdist3 is causing lots of issues so it is now optional
FULL_DEPENDENCIES = [
    "tcrdist3",
    "pwseqdist"
        ]

opts = dict(
    name=NAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
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
    extras_require={
        "dev": DEV_DEPENDENCIES,
        "full": FULL_DEPENDENCIES
    }
)


if __name__ == "__main__":
    setup(**opts)
