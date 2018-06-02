Last Updated: March 15, 2018
Author: Joseph Jaspers

BlackCat_Tensors (BCT) is a highly optimized Matrix library designed for NeuralNetworks.


Current Work:
	Focused on integrating standard Linear Algebra routines (IE BLAS routines).
	Focused on adding convolution/correlation kernel for ConvNets. 

Intallation/Setup:
	
	BCT is a header only library that supports compilation with the NVCC and G++ 
		- stable with G++ 6 and 7 and NVCC (CUDA 9)
		
	Setting simply requires adding the BlackCat_Tensors3.3 your path and including "BlackCat_Tensors.h"

FAQ Fast Explanation:
	
	Can I use non primitive types? No, BCT currently only supports primitive data types.
	
	CPU multithreading? Simply link openmp
	GPU multithreading? Install CUDA 9 run with NVCC and choose the GPU mathlibrary. 

	How to choose mathlibrary?
	BC::Vector<float, BC::GPU> myVec(sz); //Allocates data on the gpu
	BC::Vector<double, BC::CPU> myVec(sz); //Allocates data on the cpu

	**Must be linked to an apropriate BLAS with a cblas_dgemm function and a cblas_sgemm function.
	**Dotproduct currently only availalbe to double, and float types.

FAQ Fast Explanation (GPU):

	Using the cpu/gpu libraries use the identical interface.

	an expression such as:
	y = m * x + b 

	will lazily evaluate the entire expression and then sound the entire function to the GPU to run. 
	data is NOT passed back and forth between the CPU and GPU, (this is more efficient) 

Supports:

	CPU Multithreaded (via openmp)
	GPU Multithreading (via CUDA) 

Optimizations:

	Standard linear operations utilize Expression-Templates 
	Dotproducts are implemented through BLAS. Currently no default option is available. 
	(recommended BLAS implementation ATLAS) 


--------------------------------------------------------------------------------------------------------

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

Methods:

**DEFINED IN BC_Internals/BC_Core/Implementation_Core/Tensor_Operations.h

	_tensor_& operator =  (const _tensor_&) 		//copy
	_tensor_& operator =  (const T scalar) 			//fill tensor with scalar value
	_tensor_  operator +  (const _tensor_&) const		//pointwise addition
	_tensor_  operator -  (const _tensor_&) const		//pointwise subtraction
	_tensor_  operator /  (const _tensor_&) const		//pointwise scalar
	_tensor_  operator *  (const _tensor_&) const		//dotproduct or pointwise multiplication if scalar
	_tensor_  operator %  (const _tensor_&) const		//pointwise multiplication
	_tensor_  operator ** (const _tensor_&) const		//pointwise multiplication (alternate)
	_tensor_& operator += (const _tensor_&)			//assign and sum
	_tensor_& operator -= (const _tensor_&)			//assign and subract
	_tensor_& operator /= (const _tensor_&)			//assign and divice
	_tensor_& operator %= (const _tensor_&)			//assign and pointwise multiply

	_tensor_ operator == (const _tensor_&)			//LAZY ASSIGNMENT (to be nested within a lazy evaluation)
	_tensor_ operator && (const _tensor_&)			//Enables combining two expressions into a single for loop (must be same size)
	
	template<class functor> _tensor_ unExpr(functor)			//Creates a custom unary_expression to be lazily evaluated
	template<class functor> _tensor_ biExpr(functor, const_tensor_&) 	//Creates a custom binary_expression to be lazily evaluated

	NOTES:
		1) _tensor_ is not an actual type, the type returned is based upon the classes used (IE Vector,Vector Matrix etc).
		2) Tensor by Scalar operations -- return the dominant tensor type (IE the non scalar type)
		3) Scalar by Tensor operations -- return the dominant tensor type (IE operation order does matter for non commutative functions)
		4) functor object needs to have a trivial constructor and the overloaded operator()(T value) (if unary) or operator()(T val1, U val2) (if binary)

**DEFINED IN BC_Internals/BC_Core/Implementation_Core/Tensor_Utility.h

	auto eval	(void)  	const		//if an expression_tensor instantly evaluate, else return reference to self
	void print	(void) 		const		//print the tensor (formatted) **calls eval if expression tensor**
	void printSparse(void) 		const 		//print the tensor (formatted) without outputing 0s (useful for images)
	void print      (int precision) const		//same but with precision p 
	void printSparse(int precision) const		//same but with precision p
	void randomize(T lowerbound, T upperbound) 	//randomize the tensor within a given range
	void fill(T value)				//fill tensor with specific value
	void zero() 					//fill with 0s
	void zeros()					//fill with 0s (alternate) 
	
	void write(ofstream&) const			//write as a single line of CSV -> First writes rank of tensor, then dimensions, then the values
	void read(ifstream&, 				//Reads a line from a csv (regardless of size),
			bool read_dimensions=true,     //if read_dimensions assumes line was written by .write() 						 
			bool overwrite_dimensions=true)	//if overwrite_dimensions, overwrites the dimensions of the tensor (only relevant if read_dimensions is true)

**DEFINED IN BC_Internals/BC_Core/TensorBase.h

	int dims() const				//returns rank 
	int size() const				//returns size
	int rows() const				//returns rows 
	int dimension(int i) const 			//returns the dimension at a given index
	int ld1() const				//returns internal row_dimension (relevant for transpose expressions, subtensors, etc)
	int ld2() const				//returns internal col_dimension (only relevant for tensors of order > 2 IE Cubes)
	int resetShape(std::vector<int>) 		//reshapes tensor, shape must be of same rank
	void print_dimensions() const			//prints the dimensions of tensor... formated: [row][col][etc]
	void print_outer_dimensions() const			//prints the internal dimensions of tensor... formated: [ld_row][ld_cols][etc]

	const auto inner_shape() const			//returns some_array_type which holds inner shape (type depedent on context)
	const auto outer_shape() const			//returns some_array_type which holds outer shape (type depedent on context)

	const auto data() const				//returns internal iterator IE expression_functor or Core/Tensor_Slice/Tensor/Scalar
	      auto data()				//returns internal iterator IE expression_functor or Core/Tensor_Slice/Tensor/Scalar

	const operator[] (int i) const 			//returns "slice" of tensor at index (IE Cube returns Matrix, Matrix returns Vector, Vector returns Scalar)
   	      operator[] (int i)			//returns "slice" of tensor at index (IE Cube returns Matrix, Matrix returns Vector, Vector returns Scalar)

	const opeartor() (int i) const 			//returns a scalar at given index
	      operator() (int i) 			//returns a scalar at given index

	const row(int i) const 				//returns a row vector (static asserts class is matrix)
	      row(int i)				//returns a row vector (static asserts class is matrix)

	const auto operator() (int... dimlist) const 	//returns a tensor slice || IE myTensor(1,2,3) equivalent to myTensor[1][2][3] 
	      auto operator() (int... dimlist)		//returns a tensor slice || IE myTensor(1,2,3) equivalent to myTensor[1][2][3] 

	static array shapeOf (const TensorBase<T>&)	//returns shape of given tensor 

**DEFINED IN BC_Internals/BC_Core/Matrix.h and Vector.h

	cosnt auto t() const		//returns a transpose expression (cannot transpose in place)
	      auto t() 			//returns a transpose expression (cannot transpose in place)
