FROM ubuntu 
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
  apt-get install gcc-7 && \
  apt-get install g++-8 && \
  apt-get install libblas-dev
CMD [cd tests && make atlas && ./blackcat_tests]
