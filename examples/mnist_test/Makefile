CXXFLAGS = -O3 -o mnist_test_exec
CUFLAGS = -O3 -o mnist_test_exec_cu  -gencode arch=compute_52,code=sm_52

CXXFILES = MNIST_Test.cpp
CUFILES  = MNIST_Test.cu

LIBS = -lcblas
ATLASLIBS = -lcblas -latlas -lf77blas
CULIBS = -lcublas

INCLUDE = ../include/ 
LIBINCLUDE = /usr/libs/  # PATH TO YOUR BLAS 
LIBATLASINCLUDE = /home/joseph/Library_Sources/ATLAS/Build/lib  # PATH TO YOUR ATLAS-BLAS LIBRARY

all:
	$(CXX) $(CXXFILES) $(LIBS) $(CXXFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE)

atlas:
	$(CXX) $(CXXFILES) $(ATLASLIBS) $(CXXFLAGS) -I $(INCLUDE) -L $(LIBATLASINCLUDE)

cuda:
	nvcc $(CUFILES) $(CULIBS) $(CXXFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE)
	
	
clean:
	rm -f prog *.o; rm mnist_test_exec; rm mnist_test_exec_cu;
