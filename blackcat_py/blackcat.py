import cppyy
import numpy as np
import os
from .__init__ import bc
_pythonized_tensor_types = {}


class expression:
	'''
		Wraps the C++ class Expression_Base<ExpressTemplate>
		Found: blackcat/tensors/expression_base.h
	'''
	def _expr_op(function):
		def decorator(self, arg):
			if isinstance(arg, expression):
				arg = arg.tensor
			return expression(function(self, arg))

		return decorator

	def __init__(self, tensor):
		self.tensor = tensor

	@_expr_op
	def __add__(self, lv):
		return self.tensor + lv

	@_expr_op
	def __sub__(self, lv):
		return self.tensor - lv

	@_expr_op
	def __mul__(self, lv):
		return self.tensor % lv

	@_expr_op
	def __div__(self, lv):
		return self.tensor / lv

	@_expr_op
	def __matmul__(self, lv):
		return self.tensor * lv

	@property
	def strides(self):
		return self.shape.inner_shape().prod()

	@property
	def shape(self):
		return self.tensor.get_shape()

class numpy_functions:
	# reference: https://numpy.org/doc/1.18/reference/arrays.ndarray.html

	def item(*args):
		return self.tensor[args].data()[0]

	def tolist():
		lst = len(self.tensorize)
		for i in range(self.size()):
			lst[i] = self.tensor.data()[i]

	# no itemset

	def tostring():
		return self.tensor.to_string()

	def sort():
		self.tensor.sort()
		return self

class ndarray(expression, numpy_functions):

	def __init__(self, shape=None, dtype=float, allocator=None, _view=None):
		if _view:
			tensor = _view
		else:
			dim = bc.dim(*list(shape))

			if not allocator:
				ttype = bc.Tensor[dim.tensor_dim, dtype]
			else:
				ttype = bc.Tensor[dim.tensor_dim, dtype, allocator]

			name = ttype.__name__
			if name not in _pythonized_tensor_types:
				ttype.__str__ = ttype.to_string
				ttype.__repr__ = ttype.__str__
				_pythonized_tensor_types[name] = ttype

			tensor = ttype(dim)

		expression.__init__(self, tensor)

	@property
	def T(self):
		return expression(_view=self.tensor.t())

	@property
	def data(self):
		return self.tensor.data()

	@property
	def strides(self):
		return self.tensor.outer_shape()

	@property
	def size(self):
		return self.tensor.size()

	@property
	def dims(self):
		return self.tensor.inner_shape()

	@property
	def alias(self):
		return self

	@alias.setter
	def alias(self, val):
		if isinstance(val, expression):
			val = val.tensor
		self.tensor.alias().__assign__(val)
		return self

	def reshape(*shape):
		return ndarray(_view=self.tensor.reshaped(*shape))

	def flatten():
		return ndarray(_view=self.tensor.flatten())

	def ravel():
		return flatten()

	def __getattr__(self, key):
		return self.tensor.__getattribute__(key)


	def __getitem__(self, index):
		if isinstance(index, slice):
			assert(slice.step is None)
			index = bc.dim(slice.start, slice.stop)
		return ndarray(_view=self.tensor.slice(index))

	def __setitem__(self, index, lv):
		if isinstance(lv, ndarray):
			lv = lv.tensor

		return self.tensor.slice(index).__assign__(lv)

	def _expr_iop(function):
		def decorator(self, arg):
			if isinstance(arg, expression):
				arg = arg.tensor
			function(self, arg)
			return self

		return decorator

	@_expr_iop
	def __iadd__(self, lv):
		self.tensor += lv


	@_expr_iop
	def __isub__(self, lv):
		self.tensor -= lv

	@_expr_iop
	def __idiv__(self, lv):
		self.tensor /= lv

	@_expr_iop
	def __imul__(self, lv):
		self.tensor /= lv


	def __imatmul__(self, lv):
		if isinstance(lv, expression):
			lv = lv.tensor
		matmul_expr = self.tensor * lv
		self.tensor.alias().__assign__(matmul_expr)
		return self

	@property
	def i(self):
		return self

	@i.setter
	def i(self, val):
		if isinstance(val, expression):
			val = val.tensor
		self.tensor.__assign__(val)

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		return self.tensor.to_string()