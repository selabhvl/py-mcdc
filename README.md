# py-mcdc
This project aims at generating test cases satisfying modified condition decision coverage (MC/DC) criterion based on reduced ordered decision diagrams (roBDDs)

## Dependencies
- python 3.8 (minimum)
- pyeda library
- Graphviz packages

## Installing Dependencies
- Install the Python3 "development" package.

  For Debian-based systems (eg Ubuntu, Mint):

  `sudo apt-get install python3-dev`

  For RedHat-based systems (eg RHEL, Centos):

   `sudo yum install python3-devel`

- Install latest release pyeda version using pip:

   `pip3 install pyeda`

- Install pyeda from the repository:
  Clone the pyeda library:

  `git clone git://github.com/cjdrake/pyeda.git`

Build:
```
python3 setup.py clean --all
python3 setup.py build
python3 setup.py install --force --user
```

## Generating BDDs and required dependencies

- Install the Graphviz packages with the following set of commands: 
    
    ```
    apt-get update
    apt-get install graphviz*
    ```
- To generate a BDD for a Boolean expression
    ```
    # python3
    >>> from pyeda.inter import *
    >>> from graphviz import Source
    >>> a, b, c, d, e= map(bddvar, 'abcde')
    >>> f=a & (~b | ~c) & d | e
    >>> gv = Source(f.to_dot())
    >>> gv.render('Example1', view=True)
    ```
## Generating MC/DC test cases from an roBDD
- Clone the this library:

  `git clone https://github.com/selabhvl/py-mcdc.git`

- Run the default (Only 42 permutations): 
  `python3 mcdc_test/pathsearch.py`
  or 
  `python3 mcdc_test/re2.py`

- Generate MC/DC test cases from the command line

  `python3 mcdc_test/pathsearch.py numberofpermutations`

  or 

  `python3 mcdc_test/mcdctestgen.py numberofpermutations`
- Example for 1000 order permutations:

  `python3 mcdc_test/mcdctestgen.py 1000`



      


