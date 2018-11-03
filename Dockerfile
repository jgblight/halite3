FROMM ubuntu:18.04

RUN apt-get -y -qq install software-properties-common python3.6 python3-pip openjdk-8-jdk curl git
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
RUN curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
RUN apt-get update && apt-get install bazel

RUN mkdir /wheel
RUN mkdir /build
WORKDIR /build
RUN git clone https://github.com/tensorflow/tensorflow.git
WORKDIR /build/tensorflow
RUN git checkout r1.11
