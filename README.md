
# Saphyra

[![Build Status](https://travis-ci.org/jodafons/saphyra.svg?branch=master)](https://travis-ci.org/github/jodafons/saphyra)
[![PyPI Version](https://img.shields.io/pypi/v/saphyra)](https://pypi.org/project/saphyra/)
[![Python Versions](https://img.shields.io/pypi/pyversions/saphyra)](https://github.com/jodafons/saphyra)
[![License](https://img.shields.io/github/license/jodafons/saphyra)](https://github.com/jodafons/saphyra)


The machine learning package for the LPS High Energy Physics (HEP) projects. This framework uses the 
TensorFlow as core to call the machine learning training functions. Please, check this example to 
build your tuning job and launch.

## Disclaimer:

For `tensorflow` is, to this date, supported until Python 3.7 (read [here](https://github.com/tensorflow/tensorflow/issues/33374)) and
latest version of `sklearn` requires Python 3.6 or later, currently Saphyra supports Python 3.6 and 3.7 only. Any other custom setup
isn't guaranteed to work.

## Requirements:

- tensorflow;
- keras;
- numpy;
- python;
- Gaugi;
- sklearn;


## Installation using pip:


```bash
pip3 install --upgrade saphyra
```
**NOTE**: Make sure that you are using the latest package in case you have the saphyra installed. 

## Installing from source:

* Clone this repository:

```bash
git clone https://github.com/jodafons/saphyra
```

* Install it!

```bash
cd saphyra && easy_install --user .
```

## Docker:

```bash
docker run --network host -v $PWD:$HOME -it jodafons/saphyra:base
```


## Contribution:

- Dr. João Victor da Fonseca Pinto, UFRJ/COPPE, CERN/ATLAS (jodafons@lps.ufrj.br) [maintainer, developer]
- Dr. Werner Freund, UFRJ/COPPE, CERN/ATLAS (wsfreund@lps.ufrj.br) [developer]
- Msc. Micael Veríssimo de Araújo, UFRJ/COPPE, CERN/ATLAS (mverissi@lps.ufrj.br) [developer]
- Eng. Gabriel Milan Gazola, UFRJ/COPPE, CERN/ATLAS (gabriel.milan@lps.ufrj.br) [developer]
