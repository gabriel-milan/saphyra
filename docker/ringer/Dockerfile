FROM jodafons/saphyra:base

LABEL maintainer "Gabriel Gazola Milan <gabriel.milan@lps.ufrj.br>, João Victor da Fonseca Pinto <jodafons@lps.ufrj.br>, Micael Verissimo de Araujo <micael.verissimo@lps.ufrj.br>"

ENV LC_ALL C.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV TERM screen

USER root
SHELL [ "/bin/bash", "-c" ]

RUN cd /opt && git clone https://github.com/jodafons/orchestra.git && cd orchestra && pip3 install .
COPY run_tuning.py /run_tuning.py

ENTRYPOINT ["maestro.py"]
