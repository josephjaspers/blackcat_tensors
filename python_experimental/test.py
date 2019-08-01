
from blackcat_tensors import *
 
# try 
m = Matrix(3,3)
print(m)

print('m[2] = 123')
m[2] = 123
print(m)

print('try matmul' )
m += m *m

m.zero()
m.fill(4) 

print(m.__repr__())
print('m+=m')
m += m
print(m.__repr__())


print('m + m')
print(m + m)

print('m+= m + m ')
m += m + m
print(m.__repr__())


print('m slice')
print(m[0])


print('rnaged slize')
print(m[0:2])

def bench():
	sz = 10000

	import numpy as np
	a1 = np.ndarray(shape=[sz*sz], dtype=np.float64)
	b1 = np.ndarray(shape=[sz*sz], dtype=np.float64)
	c1 = np.ndarray(shape=[sz*sz], dtype=np.float64)
	d1 = np.ndarray(shape=[sz*sz], dtype=np.float64)
	for i,array in enumerate([a1,b1,c1,d1]):
		array.fill(i)
	print(a1.shape)
	print(a1.size)
	

	import time
	t0 = time.perf_counter()
	a1 += b1 + c1 - d1 / b1
	t1 = time.perf_counter()
	print('numpy time: (operation == "a += b + c - d / b" )| time taken : ' , t1 - t0) 
	
	a = Matrix(sz, sz)
	b = Matrix(sz, sz)
	c = Matrix(sz, sz)
	d = Matrix(sz, sz)
	for i,array in enumerate([a,b,c,d]):
		array.fill(i)

	import time
	t0 = time.perf_counter()
	a += b + c - d / b
	t1 = time.perf_counter()
	print('blackcat_tensors time: (operation == "a += b + c - d / b" )| time taken : ' , t1 - t0) 


bench()
bench()
bench()
