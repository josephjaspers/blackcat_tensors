from .cppyy_initialization import Vector, Matrix, Cube, bc, __setattributes
import cppyy

cppyy.cppdef("""
template<class T> 
T* integer_to_pointer(std::ptrdiff_t integer_ptr) {
	return  reinterpret_cast<T*>(integer_ptr); 
}
""")
integer_to_pointer = cppyy.gbl.integer_to_pointer

def numpy_get_ptr(np_array):
	pointer, read_only_flag = np_array.__array_interface__['data']
	return pointer 

def from_np(np_array):
	dimension = len(np_array.shape)
	array_slice = bc.tensors.exprs.Array_Slice
	allocator_t = bc.Allocator[bc.host_tag, 'double'] 
	slice_type  = array_slice[dimension, 'double', allocator_t]
	tensor_type = bc.Tensor_Base[slice_type]
	
	shape  = bc.Shape[dimension](*np_array.shape)
	np_ptr = numpy_get_ptr(np_array)
	c_ptr  = integer_to_pointer['double'](np_ptr)
	tensor = tensor_type(c_ptr, shape)

	# Fix __getitem__ __setitem__ etc
	__setattributes(type(tensor))

	# Ensure the np array stays in scope 
	setattr(tensor, 'np', np_array) 
	return tensor

def to_np(blackcat_tensor):
	raise Exception("to_np not implemented")


#import numpy as np 
#a = np.ndarray([10], dtype=float)
#a.fill(123) 
#print(a)
#a[3] = 10
#a[5] =99
#v = from_np(a) 
#print(v)
#print(v.np)

