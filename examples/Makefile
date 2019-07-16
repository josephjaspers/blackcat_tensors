CXXFLAGS = -O3 -o mnist_test_exec
CUFLAGS = -O3 -o mnist_test_exec_cu  -gencode arch=compute_52,code=sm_52

CXXFILES = MNIST_Test.cpp
CUFILES  = MNIST_Test.cu

LIBS = -lcblas -latlas -lf77blas
CULIBS = -lcublas

INCLUDE = ../include/
LIBINCLUDE = /home/joseph/Library_Sources/ATLAS/Build/lib

all:
	$(CXX) $(CXXFILES) $(LIBS) $(CXXFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE)

cuda:
	nvcc $(CUFILES) $(CULIBS) $(CXXFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE)
	
	
clean:
	rm -f prog *.o; rm mnist_test; rm mnist_test_cu;
