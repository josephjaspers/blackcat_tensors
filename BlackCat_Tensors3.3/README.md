Last Updated: Sunday, June 3rd, 2018
Author: Joseph Jaspers

BlackCat_Tensors (BCT) is a highly optimized Matrix library designed for NeuralNetworks in a multi-threaded context. Various operations enable utilizing low-level multithreaded concepts in this much higher-level framework.

############GPU LIBRARY CURRENTLY DOES NOT WORK-- WILL FIX SOON---- (nvcc does not support if-constexpr)
###########STABLE BRANCH DOES WORK FOR GPU, Though the CPU version will be faster for the current branch (master)
Current Work:
	Scaleable injection for BLAS routines, implementating convolution/correlation (using MKL/CUDA)

Intallation/Setup:
	
	BCT is a header only library that supports compilation with the NVCC and G++ 
		- stable with G++ 6 and 7 and NVCC (CUDA 9.2)
		
	Setting simply requires adding the BlackCat_Tensors3.3 your path and including "BlackCat_Tensors.h"

FAQ Fast Explanation:
	
	All internal-types are now supported, though the classes are designed for primitive-numeric types
	
	CPU multithreading? Simply link openmp
	GPU multithreading? Install CUDA 9 run with NVCC and choose the GPU mathlibrary. 

	How to choose mathlibrary?
	BC::Vector<float, BC::GPU> myVec(sz); //Allocates internal on the gpu
	BC::Vector<double, BC::CPU> myVec(sz); //Allocates internal on the cpu

	**Must be linked to an apropriate BLAS with cblas_dgemm function and cblas_sgemm function.
	**Dotproduct currently only available to double, and float types.

FAQ Fast Explanation (GPU):

	Using the cpu/gpu libraries use the identical interface.

	an expression such as:
	y = m * x + b 

	will lazily evaluate the entire expression and then sound the entire function to the GPU to run. 
	internal is NOT passed back and forth between the CPU and GPU, (this is more efficient) 

Supports:

	CPU Multithreaded (via openmp)
	GPU Multithreading (via CUDA) 

Optimizations:

	***BLAS_DETECTION and INJECTION*** (new feature)
	BlackCat_Tensor's now supports an injection system for gemm to reduce the generation of temporaries. 
	Operations utilizing matrix multiplication in which possible "injections" are available will not use a temporary to store the result of a matrix-mul operation.

	This system detects complicated matrix-mul problems such as:
		ForwardPropagation;	y = g(w * x + b)  
		In equation, the "optimal" hardcoded method would be to call a BLAS function
		evaluating w*x to y. And then calling a function such as y = g(y + b)
		This library detects this entire expression and does the exact stated strategy.

		injection may not be used with /= and %= (%= is the 'array-product' and assign)
	
	This however assumes that no aliases of the give problem are utilized. IE 	
		a = a * b; 
		will NOT result in the correct output, as it is the equivalent of a single BLAS call. 
		to utilize expressions such as these use

		a.alias() = a * b;
		and this will cause a * b to evaluate to a temporary and than copy the values to a. 
	

	All linear or O(n) operations utilize expression-templates/lazy evaluation system.
	Dotproducts are implemented through BLAS. Currently no default option is available. 
	(recommended BLAS implementation ATLAS) 


Benchmarks:
	See bottom of file
Methods:

**DEFINED IN BC_Internals/BC_Array/Implementation_Array/Tensor_Operations.h

	_tensor_& operator =  (const _tensor_&) 		//copy
	_tensor_& operator =  (const T scalar) 			//fill tensor with scalar value
	_tensor_  operator +  (const _tensor_&) const		//pointwise addition
	_tensor_  operator -  (const _tensor_&) const		//pointwise subtraction
	_tensor_  operator /  (const _tensor_&) const		//pointwise scalar
	_tensor_  operator *  (const _tensor_&) const		//dotproduct or pointwise multiplication if scalar
	_tensor_  operator %  (const _tensor_&) const		//pointwise multiplication
	_tensor_& operator += (const _tensor_&)			//assign and sum
	_tensor_& operator -= (const _tensor_&)			//assign and subract
	_tensor_& operator /= (const _tensor_&)			//assign and divice
	_tensor_& operator %= (const _tensor_&)			//assign and pointwise multiply

	//Comparisons -- each operation does an element-by-element comparison and returns the apropriate value based on the operand
	_tensor_ operator == (const _tensor_&) const		//expression of equality of tensor,               1 == true, 0 == false
	_tensor_ operator >  (const _tensor_&) const		//expression of greater-than of tensor,           1 == true, 0 == false
	_tensor_ operator <  (const _tensor_&) const		//expression of less-than of tensor,              1 == true, 0 == false
	_tensor_ operator >= (const _tensor_&) const		//expression of greater-than-or-equal of tensor,  1 == true, 0 == false
	_tensor_ operator <= (const _tensor_&) const		//expression of less-than-or-equal of tensor,  	  1 == true, 0 == false

	template<class functor> _tensor_ un_expr(functor)			//Creates a custom unary_expression to be lazily evaluated
	template<class functor> _tensor_ bi_expr(functor, const_tensor_&) 	//Creates a custom binary_expression to be lazily evaluated

	NOTES:
		1) _tensor_ is not an actual type, the type returned is based upon the classes used (IE Vector,Vector Matrix etc).
		2) Tensor by Scalar operations -- return the dominant tensor type (IE the non scalar type)
		3) Scalar by Tensor operations -- return the dominant tensor type (IE operation order does matter for non commutative functions)
		4) functor object needs to have a trivial constructor and the overloaded operator()(T value) (if unary) or operator()(T val1, U val2) (if binary)

**DEFINED IN BC_Internals/BC_Array/Implementation_Array/Tensor_Utility.h

	void print	(void) 		const		//print the tensor (formatted) **calls eval if expression tensor**
	void printSparse(void) 		const 		//print the tensor (formatted) without outputing 0s (useful for images)
	void print      (int precision) const		//same but with precision p 
	void printSparse(int precision) const		//same but with precision p
	void randomize(T lowerbound, T upperbound) 	//randomize the tensor within a given range
	void fill(T value)				//fill tensor with specific value
	void zero() 					//fill with 0s
	
	void write(ofstream&) const			//write as a single line of CSV -> First writes rank of tensor, then dimensions, then the values
	void read(ifstream&, 				//Reads a line from a csv (regardless of size),
			bool read_dimensions=true,     //if read_dimensions assumes line was written by .write() 						 
			bool overwrite_dimensions=true)	//if overwrite_dimensions, overwrites the dimensions of the tensor (only relevant if read_dimensions is true)

**DEFINED IN BC_Internals/BC_Array/Tensor_Base.h

	int dims() const				//returns number of dimensions (scalar = 0, vector = 1, matrix = 2, etc)
	int size() const				//returns number of elements 
	int rows() const				//returns rows 
	int cols() const				//returns cols 
	int dimension(int i) const 			//returns the dimension at a given index
	int ld_dimension(int i) const 			//returns the leading dimension at a given index
	int ld1() const					//returns internal row_dimension (relevant for transpose expressions, subtensors, etc)
	int ld2() const					//returns internal matrix_dimension (only relevant for tensors of order > 2 IE Cubes)
	void print_dimensions() const			//prints the dimensions of tensor... formated: [row][col][etc]
	void print_leading_dimensions() const		//prints the internal dimensions of tensor... formated: [ld_row][ld_cols][etc]

	const auto inner_shape() const			//returns some_array_type which holds inner shape (type depedent on context)
	const auto outer_shape() const			//returns some_array_type which holds outer shape (type depedent on context)

	const auto internal() const				//returns internal iterator IE expression_functor or Array/Tensor_Slice/Tensor/Scalar
	      auto internal()				//returns internal iterator IE expression_functor or Array/Tensor_Slice/Tensor/Scalar


**DEFINED IN BC_Internals/BC_Array/Tensor_Shapinh.h

	const operator[] (int i) const 			//returns "slice" of tensor at index (IE Cube returns Matrix, Matrix returns Vector, Vector returns Scalar)
   	      operator[] (int i)			//returns "slice" of tensor at index (IE Cube returns Matrix, Matrix returns Vector, Vector returns Scalar)
	const slice 	 (int i) const 			//same as operator[]
   	      slice	 (int i)			//same as operator[]
	

	const opeartor() (int i) const 			//returns a scalar at given index
	      operator() (int i) 			//returns a scalar at given index

	const row(int i) const 				//returns a row vector (static asserts class is matrix)
	      row(int i)				//returns a row vector (static asserts class is matrix)

	const col(int i) const 				//returns a slice of a matrix (same as operator[] only, available to matrices)
	      col(int i)				//returns a slice of a matrix (same as operator[] only, available to matrices)

	const auto operator() (int... dimlist) const 	//returns a tensor slice || IE myTensor(1,2,3) equivalent to myTensor[1][2][3] 
	      auto operator() (int... dimlist)		//returns a tensor slice || IE myTensor(1,2,3) equivalent to myTensor[1][2][3] 

	void resize(ints...)				//resizes the tensor and deletes the old-contents

	static _tensor_ reshape(_tensor_&)(integers...)	//reshapes the tensor to the given dimensions, this is a lazy expression
							//reshape does NOT modify the original shape, any modifications of the internal
							//effects its original source 
							//This function is curried IE reshape(myVec)(5,5) //returns vec expression reshaped to a 5x5 matrix
	static _tensor_ chunk(_tensor_&)(location_ints...)(shape_ints...) //returns a chunk of a tensor at give location integers
									  //with the respetive shape given from shape_ints...
									  //any modifications of the new internal effect its original source
									  //This function is curreid IE chunk(myCube)(2,1,0)(2,2)
									  // --- returns a 2x2matrix at page 3,column 2, row 0.

**DEFINED IN BC_Internals/BC_Array/Matrix.h and Vector.h

	cosnt auto t() const		//returns a transpose expression (cannot transpose in place)
	      auto t() 			//returns a transpose expression (cannot transpose in place)


--------------------------------------------------------------------------------------------------------

***Planning on more benchmarks soon***

Benchmarks:
https://github.com/josephjaspers/BlackCat_Libraries/blob/master/BlackCat_Tensors3.3/UnitTests/Benchmarks/BenchmarkEigen.h

(CPU only)
G++ 7
03 Optimizations
BLAS implementation: ATLAS 

Benchmarks:

	BENCHMARKING - 03 OPTIMIZATIONS, NO OPENMP
	Benchmarking: a = b + c * 3 * d - e 
	SIZE = [4][4]        Reps = 100000      BLACKCAT_TENSORS_TIME:  0.027921        EIGEN TIME: 0.021700        Eigen better by 0.006220
	SIZE = [8][8]        Reps = 100000      BLACKCAT_TENSORS_TIME:  0.051630        EIGEN TIME: 0.033965        Eigen better by 0.017666
	SIZE = [16][16]      Reps = 10000       BLACKCAT_TENSORS_TIME:  0.021360        EIGEN TIME: 0.010734        Eigen better by 0.010626
	SIZE = [64][64]      Reps = 10000       BLACKCAT_TENSORS_TIME:  0.512677        EIGEN TIME: 0.633116        Blackcat_Tensors better_by 0.120439
	SIZE = [128][128]    Reps = 1000        BLACKCAT_TENSORS_TIME:  0.371462        EIGEN TIME: 0.446243        Blackcat_Tensors better_by 0.074781
	SIZE = [256][256]    Reps = 1000        BLACKCAT_TENSORS_TIME:  2.860204        EIGEN TIME: 3.554006        Blackcat_Tensors better_by 0.693802
	SIZE = [512][512]    Reps = 100         BLACKCAT_TENSORS_TIME:  2.183934        EIGEN TIME: 2.788366        Blackcat_Tensors better_by 0.604432

	Benchmarking: a = b + c + d + e
	SIZE = [4][4]        Reps = 100000      BLACKCAT_TENSORS_TIME:  0.000977        EIGEN TIME: -0.001234       Eigen better by 0.002211
	SIZE = [8][8]        Reps = 100000      BLACKCAT_TENSORS_TIME:  0.000597        EIGEN TIME: 0.004261        Blackcat_Tensors better_by 0.003665
	SIZE = [16][16]      Reps = 10000       BLACKCAT_TENSORS_TIME:  0.003078        EIGEN TIME: 0.001818        Eigen better by 0.001260
	SIZE = [64][64]      Reps = 10000       BLACKCAT_TENSORS_TIME:  0.025323        EIGEN TIME: 0.029654        Blackcat_Tensors better_by 0.004331
	SIZE = [128][128]    Reps = 1000        BLACKCAT_TENSORS_TIME:  0.019406        EIGEN TIME: 0.019143        Eigen better by 0.000264
	SIZE = [256][256]    Reps = 1000        BLACKCAT_TENSORS_TIME:  0.065807        EIGEN TIME: 0.064978        Eigen better by 0.000830
	SIZE = [512][512]    Reps = 100         BLACKCAT_TENSORS_TIME:  0.058634        EIGEN TIME: 0.061161        Blackcat_Tensors better_by 0.002528

 	success  main

	--------------------------------------------------------------------------------------------------------

	BENCHMARKING - 03 OPTIMIZATIONS, WITH OPENMP
	Benchmarking: a = b + c * 3 * d - e 
	SIZE = [4][4]        Reps = 100000      BLACKCAT_TENSORS_TIME:  0.031624	EIGEN TIME: 0.016736	Eigen better by 0.014888
	SIZE = [8][8]        Reps = 100000      BLACKCAT_TENSORS_TIME:  0.050092	EIGEN TIME: 0.040403	Eigen better by 0.009689
	SIZE = [16][16]      Reps = 10000       BLACKCAT_TENSORS_TIME:  0.020177	EIGEN TIME: 0.017348	Eigen better by 0.002829
	SIZE = [64][64]      Reps = 10000       BLACKCAT_TENSORS_TIME:  0.541889	EIGEN TIME: 0.343337	Eigen better by 0.198552
	SIZE = [128][128]    Reps = 1000        BLACKCAT_TENSORS_TIME:  0.418615	EIGEN TIME: 0.281595	Eigen better by 0.137019
	SIZE = [256][256]    Reps = 1000        BLACKCAT_TENSORS_TIME:  2.927418	EIGEN TIME: 2.181689	Eigen better by 0.745729
	SIZE = [512][512]    Reps = 100         BLACKCAT_TENSORS_TIME:  2.141044	EIGEN TIME: 1.550337	Eigen better by 0.590707	
	Benchmarking: a = b + c + d + e
	SIZE = [4][4]        Reps = 100000      BLACKCAT_TENSORS_TIME:  0.005111	EIGEN TIME: 0.002886	Eigen better by 0.002226
	SIZE = [8][8]        Reps = 100000      BLACKCAT_TENSORS_TIME:  0.005168	EIGEN TIME: 0.000525	Eigen better by 0.004643
	SIZE = [16][16]      Reps = 10000       BLACKCAT_TENSORS_TIME:  -0.000004	EIGEN TIME: -0.001399	Eigen better by 0.001395
	SIZE = [64][64]      Reps = 10000       BLACKCAT_TENSORS_TIME:  0.022954	EIGEN TIME: 0.027806	Blackcat_Tensors better_by 0.004852
	SIZE = [128][128]    Reps = 1000        BLACKCAT_TENSORS_TIME:  0.005003	EIGEN TIME: 0.012548	Blackcat_Tensors better_by 0.007545
	SIZE = [256][256]    Reps = 1000        BLACKCAT_TENSORS_TIME:  0.036842    EIGEN TIME: 0.063661	Blackcat_Tensors better_by 0.026819
	SIZE = [512][512]    Reps = 100         BLACKCAT_TENSORS_TIME:  0.050181	EIGEN TIME: 0.063029	Blackcat_Tensors better_by 0.012848
 	success  main

	Notes:
	Benchmark 2 used ATLAS dgemm (no multithreading) performance should be comparable or better with apropriate benchmark
	Benchmark to be added in future 
	If these benchmarks seem misleading/incorrect please post an issue, and I will review accordingly.
	-------------------------------------------------------------------------------------------------------

