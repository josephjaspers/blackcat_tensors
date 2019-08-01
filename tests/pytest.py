import cppyy

import os
os.environ['OMP_NUM_THREADS'] = '8'
cppyy.cppdef('#include <omp.h>') 
cppyy.load_library('gomp')
cppyy.gbl.omp_set_num_threads(8)
#cppyy.cppdef('omp_set_num_threads(4);') 

cppyy.cppdef('#define BC_BLAS_USE_C_LINKAGE') 
cppyy.load_library('/usr/lib/libblas.so')
cppyy.cppdef('#include "run.h"')

cppyy.gbl.BC.tests.run() 
