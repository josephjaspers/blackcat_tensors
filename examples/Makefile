CXXFLAGS = -O3 -o mnist_test_exec
CXXFILES = MNIST_Test.cpp
LIBS = -lcblas -latlas -lf77blas
INCLUDE = ../include/
LIBINCLUDE = /home/joseph/Library_Sources/ATLAS/Build/lib

all:
	$(CXX) $(CXXFILES) $(LIBS) $(CXXFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE)
	
clean:
	rm -f prog *.o; rm mnist_test_exec