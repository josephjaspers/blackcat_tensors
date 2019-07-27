CXXFLAGS = -O3 -o blackcat_tensors_test
CUFLAGS = -O3 -o blackcat_tensors_test_cu

CXXFILES = run.cpp
CUFILES  = run.cu

LIBS = -lcblas
ATLASLIBS = -lcblas -latlas -lf77blas
CULIBS = -lcublas

INCLUDE = ../include/  -I /usr/include/
LIBINCLUDE = /usr/lib/

default:
	$(CXX) $(CXXFILES) $(LIBS) $(CXXFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE)

atlas:
	$(CXX) $(CXXFILES) $(ATLASLIBS) $(CXXFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE)

cuda:
	nvcc $(CUFILES) $(CULIBS) $(LIBS) $(CUFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE)

cuda_atlas:
	nvcc $(CUFILES) $(CULIBS) $(ATLASLIBS) $(CUFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE)

	
clean:
	rm -i prog *.o; rm blackcat_tensors_test; rm blackcat_tensors_test_cu;
