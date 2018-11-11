FROM ubuntu:18.04

RUN apt-get update && apt-get -y -qq install software-properties-common python3-dev python3-pip openjdk-8-jdk curl git
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN apt-get update && apt-get -y -qq install bazel

RUN ln -sfn /usr/bin/python3.6 /usr/bin/python

RUN mkdir /wheel
RUN mkdir /build
WORKDIR /build
RUN git clone https://github.com/tensorflow/tensorflow.git
WORKDIR /build/tensorflow
RUN git checkout r1.11

RUN pip3 install numpy keras_applications keras_preprocessing wheel
