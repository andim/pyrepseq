# Contributing

You are cordially invited to contribute to the further development of pyrepseq! There are various ways of contributing (improving documentation, fixing bugs, and extending functionality) all of which are very welcome.

Simply hack away and open a Pull request. Alternatively open an issue if you want to discuss your idea first.

## Installing `pyrepseq` in a development environment

You will want to install `pyrepseq` from the latest version of the source code.
To do this, first use a tool like `venv` or `conda` to create a new development environment, and activate it.
Then, clone this repository, and from within the project root directory, run:

```
$ pip install -e ".[dev]"
```

### What this is doing
1. By `pip install`ing the current directory (`.`), you are install `pyrepseq` from the source code you just cloned, and not from the PyPI server.
2. You will want to directly link the `pyrepseq` source code you are editing to your environment's install of `pyrepseq`, so that all changes you make to the source code are reflected in real time in your install. The "editable" flag (`-e`) ensures that this is the case.
3. You will want to install some development dependencies along with `pyrepseq` itself, such as `pytest`, `sphinx`, etc. This is done by specifying the `dev` extras group.

## Unit testing

The module comes with a number of unit tests to increase robustness. To make sure your new code does not break any existing functionality, you can run the unittest by typing `pytest` in the top-level directory.
