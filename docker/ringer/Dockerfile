FROM jodafons/saphyra:base

LABEL maintainer "Gabriel Gazola Milan <gabriel.milan@lps.ufrj.br>, João Victor da Fonseca Pinto <jodafons@lps.ufrj.br>, Micael Verissimo de Araujo <micael.verissimo@lps.ufrj.br>"

ENV LC_ALL C.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV TERM screen

USER root
SHELL [ "/bin/bash", "-c" ]

COPY run_tuning.py /run_tuning.py
RUN chmod 774 /run_tuning.py

ENTRYPOINT ["/run_tuning.py"]
