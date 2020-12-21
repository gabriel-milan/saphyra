


[![Build Status](https://travis-ci.org/jodafons/saphyra.svg?branch=master)](https://travis-ci.org/github/jodafons/saphyra)
[![PyPI Version](https://img.shields.io/pypi/v/saphyra)](https://pypi.org/project/saphyra/)
[![Python Versions](https://img.shields.io/pypi/pyversions/saphyra)](https://github.com/jodafons/saphyra)

# Saphyra

In 2017 the ATLAS experiment implemented an ensemble of neural networks (NeuralRinger algorithm) dedicated to improving the performance of filtering events containing electrons in the high-input rate online environment of the Large Hadron Collider at CERN, Geneva. The ensemble employs a concept of calorimetry rings. The training procedure and final structure of the ensemble are used to minimize fluctuations from detector response, according to the particle energy and position of incidence. This reposiroty is dedicated to hold all analysis scripts for each subgroup in the ATLAS e/g trigger group.

## What is it for?

Saphyra is a package used to derive all ringer models in large scale using HPC cluster infrastruture.

**NOTE** This repository make part of the ringer derivation kit.

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
cd saphyra && easy_install --user .
```

## Disclaimer:

For `tensorflow` is, to this date, supported until Python 3.7 (read [here](https://github.com/tensorflow/tensorflow/issues/33374)) and
latest version of `sklearn` requires Python 3.6 or later, currently Saphyra supports Python 3.6 and 3.7 only. Any other custom setup isn't guaranteed to work.




