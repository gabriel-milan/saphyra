

[![PyPI Version](https://img.shields.io/pypi/v/saphyra)](https://pypi.org/project/saphyra/)
[![Python Versions](https://img.shields.io/pypi/pyversions/saphyra)](https://github.com/jodafons/saphyra)

# saphyra

We should include some description here.

**NOTE** This repository make part of the ringer derivation kit (rtk).

## What is it for?

Saphyra is a package used to derive all ringer models in large scale using HPC cluster infrastruture. For LPS cluster you can use the [orchestra](https://github.com/jodafons/orchestra.git) infrastruture to launch your jobs into the slurm queue.

## Installation:

Please follow these instructions below to install the saphyra package into your system.

### Installing from pip:

```bash
pip3 install --upgrade saphyra
```
**NOTE**: Make sure that you are using the latest package in case you have the saphyra installed. 

### Installing from source:

Clone this repository:
```bash
git clone https://github.com/jodafons/saphyra
```
And install it!
```bash
cd saphyra && source scripts/setup.sh
```

## Build status:

|  Branch    | Build Status |
| ---------- | ------------ |
|   Master   |[![Build Status](https://travis-ci.org/jodafons/saphyra.svg?branch=master)](https://travis-ci.org/github/jodafons/saphyra)|

## Disclaimer:

For `tensorflow` is, to this date, supported until Python 3.7 (read [here](https://github.com/tensorflow/tensorflow/issues/33374)) and
latest version of `sklearn` requires Python 3.6 or later, currently Saphyra supports Python 3.6 and 3.7 only. Any other custom setup isn't guaranteed to work.




