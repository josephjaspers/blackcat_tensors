FROM ubuntu 
ENV DEVIAN_FRONTEND
RUN apt-get update && \
  apt-get install gcc-7 && \
  apt-get install g++-8 && \
  apt-get install libblas-dev
CMD [cd examples && make atlas && ./blackcat_tests]
  
  
