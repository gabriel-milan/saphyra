FROM tensorflow/tensorflow:latest-gpu-py3

LABEL maintainer "Gabriel Gazola Milan <gabriel.milan@lps.ufrj.br>, João Victor da Fonseca Pinto <jodafons@lps.ufrj.br>, Micael Verissimo de Araujo <micael.verissimo@lps.ufrj.br>"

ENV LC_ALL C.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV TERM screen

USER root
SHELL [ "/bin/bash", "-c" ]

RUN apt-get update
RUN apt-get install -y git vim
RUN python3 -m pip install -U pip
RUN python3 -m pip install \
  sklearn \
  keras \
  sqlalchemy \
  numpy \
  kubernetes \
  psycopg2-binary \
  benedict \
  flask-login \
  saphyra \
  kolmov \
  gast==0.2.2 \
  tensorboard==2.1.1 \
  tensorflow-estimator==2.1.0 \
  tensorflow-gpu==2.1.0 \
  tensorflow==2.1.0