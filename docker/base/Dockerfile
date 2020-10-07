FROM tensorflow/tensorflow:2.3.1-gpu

LABEL maintainer "Gabriel Gazola Milan <gabriel.milan@lps.ufrj.br>, João Victor da Fonseca Pinto <jodafons@lps.ufrj.br>, Micael Verissimo de Araujo <micael.verissimo@lps.ufrj.br>"

ENV LC_ALL C.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV TERM screen

USER root
RUN apt-get install -y git vim
RUN pip install saphyra tensorflow-gpu==2.3.1 keyrings.alt 


SHELL [ "/bin/bash", "-c" ]
