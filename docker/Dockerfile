FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

RUN apt-get update && apt-get install -y apt-utils software-properties-common build-essential
RUN apt-get install -y cmake-qt-gui python3

RUN apt-get -y install python3-pip

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade setuptools

COPY ../requirements.txt ./

RUN pip3 install -r requirements.txt
