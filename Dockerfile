FROM ubuntu 
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y && \
  apt-get -y install gcc-7 && \
  apt-get -y install g++-8 && \
  apt-get -y install libblas-dev
CMD [cd tests && make atlas && ./blackcat_tests]
