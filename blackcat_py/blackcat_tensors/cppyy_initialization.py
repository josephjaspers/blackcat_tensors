# Attempt to initialize with cpu-multithreading 
import os
#os.environ['EXTRA_CLING_ARGS'] = '-fopenmp'

# Enable avx/openmp in cppyy 
#import cppyy_backend.loader as cploader 
#cploader.set_cling_compile_options(True)

# Default cppyy initialization
import cppyy
cppyy.cppdef('#pragma cling optimize 3 ')
cppyy.cppdef('#define BC_BLAS_USE_C_LINKAGE=1 ') 
cppyy.cppdef('#define BC_CLING_JIT=1 ') # defines using cling in BlackCat_Tensors.h 
cppyy.load_library('/usr/lib/libblas.so')


#try: 
#	cppyy.load_library('gomp') # TODO allow user to specify library name 
#	cppyy.gbl.omp_set_num_threads(8)	
#except:
#	print('Failure to load library "gomp", elementwise-operations will be singlethreaded') 

#Import BlackCat_Tensors.h relative to this file 
from pathlib import Path
dirname = Path(__file__).parents[2]
bct_header = os.path.join(dirname, 'include','BlackCat_Tensors.h')
cppyy.cppdef('#include "{header}"'.format(header=bct_header))

#import blackcat neuralnetworks
bct_header = os.path.join(dirname, 'include','BlackCat_NeuralNetworks.h')
cppyy.cppdef('#include "{header}"'.format(header=bct_header))

cppyy.cppdef('#pragma cling optimize 3')
#C++ bc namespace 
bc = cppyy.gbl.BC

# Create expression_traits 'python' version
class __ExpressionTraits:
	def __getitem__(self, cls):
		return bc.tensors.exprs.expression_traits[type(cls)]
expression_traits = __ExpressionTraits() 


# Initialize the tensor classes, 
# certain methods need 'fixing' as well as needing to be propagated to slices of tensors
def __setattributes(cls):
	def __repr__(self):
		return self.to_string()

	def __str__(self):
		return self.to_string() 

	def __getitem__(self, index):
		if (expression_traits[self].is_expr):
			raise Exception("Cannot take slice of an expression")

		if isinstance(index, slice):
			slice_obj = self.slice(index.start, index.stop)
		else: 
			slice_obj = self.slice(index)

		__setattributes(type(slice_obj))
		return slice_obj

	def __setitem__(self, index, value):
		tensor_slice = self.__getitem__(index) 
		tensor_slice.assign(value)
		return tensor_slice 
	
	for attr in [__repr__, __str__, __getitem__, __setitem__]:
		setattr(cls, attr.__name__, attr)

# The default available classes TODO enable different dtypes 
# Note: -- python's 'float' type is C++'s double type. 
Cube   = bc.Cube['double']
Matrix = bc.Matrix['double']
Vector = bc.Vector['double'] 


for t in [Cube, Matrix, Vector]:
	__setattributes(t) 
