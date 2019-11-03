CXXFLAGS = -O3 -o mnist_test_exec
CUFLAGS = -O3 -o mnist_test_exec_cu --use_fast_math

CXXFILES = mnist_test.cpp
CUFILES  = mnist_test.cu

LIBS = -lblas

ATLAS_BLAS_LINKAGE = -D BC_BLAS_USE_C_LINKAGE=1
ATLASLIBS =  -lf77blas -lcblas -latlas
CULIBS = -lcublas

INCLUDE = ../include/  -I /usr/include/ -I /home/joseph/BlackCat_Tensors/BlackCat_Tensors/include/
LIBINCLUDE = /usr/lib/

default:
	$(CXX) $(CXXFILES) $(LIBS) $(CXXFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE)

atlas:
	 $(CXX) $(ATLAS_BLAS_LINKAGE) $(CXXFILES)  $(CXXFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE) $(ATLASLIBS)

TRAVIS_LIBS = -lblas
travis:
	$(CXX) $(CXXVERSION) $(CXXFILES) $(TRAVIS_LIBS) $(CXXFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE)

cuda:
	nvcc $(CUFILES) $(CULIBS) $(LIBS) $(CUFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE)

cuda_atlas:
	nvcc $(CUFILES) $(ATLAS_BLAS_LINKAGE) $(CULIBS) $(CUFLAGS) -I $(INCLUDE) -L $(LIBINCLUDE) $(ATLASLIBS)
	
clean:
	rm -i mnist_test_exec; rm -i mnist_test_exec_cu;
