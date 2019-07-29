CXXFLAGS = -O3 -o blackcat_tensors_test
CUFLAGS = -O3 -o blackcat_tensors_test_cu

CXXFILES = run.cpp
CUFILES  = run.cu

LIBS = -lcblas

ATLAS_BLAS_LINKAGE = -D BC_BLAS_USE_C_LINKAGE=1
ATLASLIBS =  -lf77blas -lcblas -latlas
CULIBS = -lcublas

INCLUDE = ../include/  -I /usr/include/
LIBINCLUDE = /usr/lib/

default:
	$(CXX) $(CXXFILES) $(LIBS) $(CXXFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE)

atlas:
	 $(CXX) $(ATLAS_BLAS_LINKAGE) $(CXXFILES)  $(CXXFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE) $(ATLASLIBS)

cuda:
	nvcc $(CUFILES) $(CULIBS) $(LIBS) $(CUFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE)

cuda_atlas:
	nvcc $(CUFILES) $(ATLAS_BLAS_LINKAGE) $(CULIBS) $(CUFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE) $(ATLASLIBS)

	
clean:
	rm -i prog *.o; rm blackcat_tensors_test; rm blackcat_tensors_test_cu;
