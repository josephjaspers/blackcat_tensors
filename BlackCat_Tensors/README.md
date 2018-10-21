Last Updated: Thursday, October 14, 2018

BlackCat_Tensors (BCT) is a highly optimized Matrix library designed for NeuralNetwork construction. 
BCT is designed to support GPU computing (CUDA) and CPU multi-threading (OpenMP).
BCT focuses on delivering a high-level framework for Neural Network construction with low-level performance. 

Current Work:
	Convolution/Correlation kernels for convNets

Intallation/Setup:
	
	BCT is a header only library that supports compilation with the NVCC and G++
	BCT does not support any default BLAS routines, and must be linked with an apropriate BLAS library. 
		
	Setting up simply requires adding the BlackCat_Tensors your path and including "BlackCat_Tensors.h"

FAQ Fast Explanation:
	
	CPU multithreading? Simply link OpenMP
	GPU multithreading? Simply link CUDA 9

	How to choose allocatorrary?
	BC::Vector<float, BC::GPU> myVec(sz); //Allocates data on the gpu
	BC::Vector<double, BC::CPU> myVec(sz); //Allocates data on the cpu

	**Must be linked to an apropriate BLAS with cblas_dgemm function and cblas_sgemm function.
	**Dotproduct currently only available to double, and float types.
	**CUDA BLAS routines only support floats. 

	Non-numeric types are supported, though non-numeric types are not heavily tested in release. 

Supports:

	CPU Multithreaded (via openmp)
	GPU Multithreading (via CUDA)

Why Use BCT Opposed to Armadillo/Eigen/XYZ_Library: 

	BCT is designed specifically for Neural Network and certain features are designed in regard to this.
	A) Blas injection:
		The expression -> y = sigmoid(w * x + b)  (this is forward propagation in feedforward layers)
		First, w*x will be evaluated to y,
		Secondly, the new expression y = sigmoid(y + b)
		will be evaluated in a single for loop.
		Therefor while evaluating this expression 0 temporaries are generated. 
		(This is a compile time feature, and incurs nearly 0 overhead).

	B) Tensor Broadcasting:
		Tensor broadcasting is the ability of BCT to handle operations of different-dimensions
		IE a 'mat += vec' expression will add the column vector to every column of the matrix.
		Tensor broadcasting is useful in batch-training when we want to utilize
		matrix multiplications operations to optimize training (which is more efficient than matrix-vector operations)

Optimizations:

	Two relevant built in optimizations are explained briefly here. 

	A) The standard expression-template system allows converting multiple element-wise operations into single for loops.
		For example an operation such as:
		y = a + b - c / d; 

		Will automatically produce assembly code that is identical to

		for (int i = 0; i < y.size(); ++i)
			y[i] = a[i] + b[i] - c[i] / d[i];

		This in short allows users to use code that appears to be identical to mathematical expressions with performance equivalent to hand written code.

		If you are interested in how expression templates are implemented, (Todd Veldhuizen is the original creator of expression-templates)
		https://www.cct.lsu.edu/~hkaiser/spring_2012/files/ExpressionTemplates-ToddVeldhuizen.pdf		

	B) BlackCat_Tensor's now supports an injection system for gemm (and other BLAS routines) to reduce the generation of temporaries. 
	Operations utilizing matrix multiplication in which possible "injections" are available will not use a temporary to store the result of a matrix-mul operation.

	This system detects complicated matrix-mul problems such as:
		For example, in the forward-propagation algorithm of NeuralNetworks
		
		y = sigmoid(w * x + b); 

		The "optimized" hard-coded form would be 

		1) evalute the product of w and x to y.
		'y = w * x' (which is evaluated via a single BLAS call gemm(y, w, x))

		2) evaluate the element-wise operation of
		'y = sigmoid (y + b)' (which is evaluated with a single for loop through the expression-template system)

		
		In another more complex example (forward propagation for recurrent neural networks)
		c = sigmoid(w * x + r * y + b)  
	
		BCT convert the problem to
		'y = w * x'	(BLAS_gemm call)
		'y += r * y'	(another BLAS_gemm call)
		'y = g(y + b)'	(finally the element-wise operation (this is a single for loop))

		Certain algorithms require temporaries. 
		IE

		y += sigmoid(w * x + b)

		must use a temporary as the sigmoid function must be evaluated before summing the output to y.

		However an equation such as
		y += w * x + b

		will be evaluated as 

		y += w * x (a BLAS_gemm call)
		y += b	   (element-wise operation)

		***Caveat***
	
		BCT cannot detect if an alias is used.
		so problems such as:
			 y = y * x
		will cause the BLAS_gemm call to start writing to Y as it is still calculating the product. 
		When reusing tensors you may use aliases to skip this optimization.
		Example: y.alias() = y * x


	All linear or O(n) operations utilize expression-templates/lazy evaluation system.
	Dotproducts are implemented through BLAS. Currently no default option is available. 


Benchmarks:
	See bottom of file
Methods:

**SOURCE https://github.com/josephjaspers/BlackCat_Tensors/blob/master/BlackCat_Tensors/BC_Internals/BC_Tensor/Tensor.h
    Tensor_Types 
    
        scalar_t  --> the scalar_type of the Tensor
        allocator_t --> BC::CPU or BC::GPU
        
        Standard Tensor Types
            Tensor<int x, scalar_t, allocator_t> 
            Scalar<scalar_t, allocator_t>
            Vector<scalar_t, allocator_t>
            Matrix<scalar_t, allocator_t>

        View Types - accepts an Array to construct. Shares the internal_pointer and is not mutable. 
            Tensor_View<int x, scalar_t, allocator_t>
            Scalar_View<scalar_t, allocator_t>
            Vector_View<scalar_t, allocator_t>
            Matrix_View<scalar_t, allocator_t>
           
        Shared Types - accepts a non-const Array to construct. Shares the internal_pointer but is mutable
            Tensor_Shared<int x, scalar_t, allocator_t> 
            Scalar_Shared<scalar_t, allocator_t> 
            Vector_Shared<scalar_t, allocator_t>
            Matrix_Shared<scalar_t, allocator_t>
        
        expression Types - any tensor that is expression or array of the apropriate dimensionality
            expr::Tensor<int> 
            expr::Scalar
            expr::Vector
            expr::Matrix
            
            



----------------------------------------------------tensor operations-------------------------------------------------------

SOURCE: 	https://github.com/josephjaspers/BlackCat_Tensors/blob/master/BlackCat_Tensors/BC_Internals/BC_Tensor/Tensor_Operations.h
	** The following options also provide support for single_scalar operations 	

	_tensor_& operator =  (const _tensor_&) 		//copy
	_tensor_& operator =  (const T scalar) 			//fill tensor with scalar value
	_tensor_  operator +  (const _tensor_&) const		//pointwise addition
	_tensor_  operator -  (const _tensor_&) const		//pointwise subtraction
	_tensor_  operator /  (const _tensor_&) const		//pointwise scalar
	_tensor_  operator *  (const _tensor_&) const		//gemm or pointwise multiplication if scalar
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

	template<class functor> _tensor_ un_expr(functor)			//Creates a custom Unary_Expression to be lazily evaluated
	template<class functor> _tensor_ bi_expr(functor, const_tensor_&) 	//Creates a custom Binary_Expression to be lazily evaluated

	NOTES:
		1) _tensor_ is not an actual type, the type returned is based upon the classes used (IE Vector,Vector Matrix etc).
		2) Tensor by Scalar operations -- return the dominant tensor type (IE the non scalar type)
		3) Scalar by Tensor operations -- return the dominant tensor type (IE operation order does matter for non commutative functions)
		4) functor object needs to have a trivial constructor and the overloaded operator()(T value) (if unary) or operator()(T val1, U val2) (if binary)




----------------------------------------------------tensor utility methods--------------------------------------------------

**SOURCE:
https://github.com/josephjaspers/BlackCat_Tensors/blob/master/BlackCat_Tensors/BC_Internals/BC_Tensor/Tensor_Utility.h

	void print      (int precision=6) const		//print the tensor (formatted)
	void printSparse(int precision=6) const		//print the tensor but ignore 0s (formatted)
	void randomize(T lowerbound, T upperbound) 	//randomize the tensor within a given range
	void fill(T value)				//fill tensor with specific value
	void zero() 					//fill with 0s
	
	void write(ofstream&) const			//write as a single line of CSV -> First writes rank of tensor, then dimensions, then the values
	void read(ifstream&, 				//Reads a line from a csv (regardless of size),
			bool read_dimensions=true,     //if read_dimensions assumes line was written by .write() 						 
			bool overwrite_dimensions=true)	//if overwrite_dimensions, overwrites the dimensions of the tensor (only relevant if read_dimensions is true)




----------------------------------------------------tensor metadata methods------------------------------------------------

**SOURCE:
https://github.com/josephjaspers/BlackCat_Tensors/blob/master/BlackCat_Tensors/BC_Internals/BC_Tensor/Expression_Templates/Shape.h

	int dims() const				//returns number of dimensions (scalar = 0, vector = 1, matrix = 2, etc)
	int size() const				//returns number of elements 
	int rows() const				//returns rows 
	int cols() const				//returns cols 
	int dimension(int i) const 			//returns the dimension at a given index
	int leading_dimension(int i) const 		//returns the leading dimension at a given index
	void print_dimensions() const			//prints the dimensions of tensor... formated: [row][col][etc]
	void print_leading_dimensions() const		//prints the internal dimensions of tensor... formated: [ld_row][ld_cols][etc]

	const auto inner_shape() const			//returns some_array_type which holds inner shape (type depedent on context)
	const auto outer_shape() const			//returns some_array_type which holds outer shape (type depedent on context)





----------------------------------------------------tensor views and data accessing-----------------------------------------

**SOURCE:
https://github.com/josephjaspers/BlackCat_Tensors/blob/master/BlackCat_Tensors/BC_Internals/BC_Tensor/Tensor_Shaping.h

	const operator[] (int i) const 			//returns "slice" of tensor at index (IE Cube returns Matrix, Matrix returns Vector, Vector returns Scalar)
   	      operator[] (int i)			//returns "slice" of tensor at index (IE Cube returns Matrix, Matrix returns Vector, Vector returns Scalar)

	const operator[] (range) const 			//returns ranged slice, similair to python's array[start:end] ---> BCT notation: array[{start, end}]
   	      operator[] (range)			//returns ranged slice, similair to python's array[start:end] ---> BCT notation: array[{start, end}]

	const slice 	 (int i) const 			//same as operator[]
   	      slice	 (int i)			//same as operator[]

	const slice	(int from, int to)		//returns a ranged slice
	      slice     (int from, int to)		//returns a ranged slice 
	
	const auto transpose() const 			//returns a transpose expression (cannot transpose in place)
	      auto transpose() 				//returns a transpose expression (cannot transpose in place)

	const auto t() const				//short hand for transpose
	      auto t() 					//short hand for transpose

	const operator() (int i) const 			//returns a scalar at given index
	      operator() (int i) 			//returns a scalar at given index

	const row(int i) const 				//returns a row vector (static asserts class is matrix)
	      row(int i)				//returns a row vector (static asserts class is matrix)

	const col(int i) const 				//returns a slice of a matrix (same as operator[], only available to matrices)
	      col(int i)				//returns a slice of a matrix (same as operator[], only available to matrices)

	const auto operator() (int... dimlist) const 	//returns a tensor slice || IE myTensor(1,2,3) comparable to myTensor[1][2][3] 
	      auto operator() (int... dimlist)		//returns a tensor slice || IE myTensor(1,2,3) comparable to myTensor[1][2][3] 


	static _tensor_ reshape(_tensor_&)(integers...)	//reshapes the tensor to the given dimensions, this is a lazy expression
							//reshape does NOT modify the original shape, any modifications of the internal
							//effects its original source 
							//This function is curried IE reshape(myVec)(5,5) //returns vec expression reshaped to a 5x5 matrix
	static _tensor_ chunk(_tensor_&)(location_ints...)(shape_ints...) //returns a chunk of a tensor at give location integers
									  //with the respetive shape given from shape_ints...
									  //any modifications of the new internal effect its original source
									  //This function is curreid IE chunk(myCube)(2,1,0)(2,2)
									  // --- returns a 2x2matrix at page 3,column 2, row 0.




----------------------------------------------------supported lazy functions------------------------------------------------

**SOURCE:
https://github.com/josephjaspers/BlackCat_Tensors/blob/master/BlackCat_Tensors/BC_Internals/BC_Tensor/Tensor_Functions.h
(implementation) https://github.com/josephjaspers/BlackCat_Tensors/blob/master/BlackCat_Tensors/BC_Internals/BC_Tensor/Expression_Templates/Operations/Unary.h
	
	Lazily_Support functions: 
	auto function_name(const _tensor_)				 //returns a lazy expression of the stated function, 

	logistic							//logistic function '1/(1 + exp(-x))'
	dx_logistic							//derivative
	cached_dx_logistic						//cached derivative
	dx_tanh								//tanh derivative
	cached_dx_tanh							//tanh cached derivative
	relu								//relu (max(0, x))
	dx_relu								//relu_derivative (x > 0 ? 1 : 0)
	cached_dx_relu							//relu_derivative (x > 0  ? 1 : 0)

	abs
	acos
	acosh
	sin
	asin
	asinh
	atan
	atanh
	cbrt
	ceil
	cos
	cosh
	exp
	exp2
	fabs
	floor
	fma
	isinf
	isnan
	log
	log2
	lrint
	lround
	modf
	sqrt
	tan
	tanh





--------------------------------------------------------------------------------------------------------

***Planning on more benchmarks soon***

Benchmarks:
https://github.com/josephjaspers/BlackCat_Libraries/blob/master/BlackCat_Tensors/UnitTests/Benchmarks/BenchmarkEigen.h

(CPU only)
G++ 7
03 Optimizations
BLAS implementation: ATLAS 

*****THESE BENCHMARKS ARE OUTDATED*****

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

