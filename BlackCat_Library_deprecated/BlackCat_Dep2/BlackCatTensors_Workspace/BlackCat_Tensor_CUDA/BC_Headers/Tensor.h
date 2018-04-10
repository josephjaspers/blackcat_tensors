#ifndef BlackCat_Tensor_super_h
#define BlackCat_Tensor_super_h

#include "HelperFunctions.h"
#include "Scalar.h"
#include "CPU.h"
#include <iostream>
#include <fstream>
#include <istream>
#include <string>
#include <sstream>
#include <initializer_list>
#include <unordered_map>
#include <map>
#include <vector>

typedef std::vector<unsigned> Shape;

template<typename number_type, class TensorOperations = CPU>
class Tensor {

	//--------------------------------------------------------Members --------------------------------------------------------//
	const bool tensor_ownership;																				//Ownership - determines if Tensor controls life span of data
	const bool rank_ownership;
	const bool ld_ownership;

	const bool subTensor;

	mutable bool needsUpdate = true;
	const Tensor<number_type, TensorOperations>* parent = nullptr;

	number_type* tensor = nullptr;																		//tensor - the actual internal array of data
	unsigned* ranks = nullptr;																			//ranks -(dimensions) an array that stores the dimensions of the Tensor
	unsigned* ld = nullptr;																				//LeadingDimension - stores data regarding the leading_dimension

	unsigned sz = 0;																					//sz - total size of data [in non subTensors same as leading_dim]
	unsigned order = 0;																					//order - the number of degree in the matrix [dimensionality]

	mutable std::unordered_map<unsigned, Tensor<number_type, TensorOperations>*> IndexTensorMap;							//index Tensor map - caches index tensors [for faster access]
	mutable std::unordered_map<std::pair<Shape, Shape>,Tensor<number_type, TensorOperations>*,BC::indexShape_pairHasher, BC::indexShape_pairEqual> SubTensorMap;	//index Tensor map - caches index tensors [for faster access]

protected:
	//--------------------------------------------------------Private Constructors--------------------------------------------------------//

	Tensor<number_type, TensorOperations>* generateIndexTensor(unsigned index) const;															//Sub_Tensor generator
	Tensor<number_type, TensorOperations>* generateSubTensor(Shape index, Shape shape) const;

	Tensor<number_type, TensorOperations>(const Tensor<number_type, TensorOperations>& m1, const Tensor<number_type, TensorOperations>& m2); 								//dot product constructor

	Tensor<number_type, TensorOperations>(const Tensor<number_type, TensorOperations>& super_tensor, unsigned index);									//Sub_Tensor constructor
	Tensor<number_type, TensorOperations>(const Tensor<number_type, TensorOperations>& super_tensor, Shape index, Shape shape);

	Tensor<number_type, TensorOperations>(const Tensor<number_type, TensorOperations>& cpy_tensor, bool copy_values);									//copy shape constructor
	Tensor<number_type, TensorOperations>(const Tensor<number_type, TensorOperations>& cpy_tensor, bool copy_values,bool transposed);					//copy shape transpose constructor


public:
	void alertUpdate() const {
		needsUpdate = true;

		if (parent)
			parent->alertUpdate();
	}

	//--------------------------------------------------------Public Constructors--------------------------------------------------------//
	Tensor<number_type, TensorOperations>();																							//Empty initalize
	Tensor<number_type, TensorOperations>(Tensor<number_type, TensorOperations> && cpy);																//move constructor
	Tensor<number_type, TensorOperations>(const Tensor<number_type, TensorOperations>& cpy);															//copy constructor
	Tensor<number_type, TensorOperations>(const Shape& shape);																		//Shape constructor
	//Other type of initializers
	Tensor<number_type, TensorOperations>(std::initializer_list<unsigned> ranks);														//scaleable shape constructor
	Tensor<number_type, TensorOperations>(unsigned m, unsigned n, unsigned k, unsigned p);
	Tensor<number_type, TensorOperations>(unsigned m, unsigned n, unsigned k);
	Tensor<number_type, TensorOperations>(unsigned m, unsigned n);

	explicit Tensor<number_type, TensorOperations>(unsigned m);
	virtual ~Tensor<number_type, TensorOperations>();

	//--------------------------------------------------------Basic Accessors--------------------------------------------------------//
	unsigned totalMatrices()					const;
	bool isInitialized() 						const;
	unsigned matrix_size() 						const;
	unsigned size() 							const;
	unsigned rows() 							const;
	unsigned cols() 							const;
	unsigned pages()							const;
	bool isMatrix() 							const;
	bool isSquare() 							const;
	bool isVector()								const;
	bool isScalar() 							const;
	unsigned rank(unsigned rank_index) 			const;
	unsigned leading_dim(unsigned rank_index) 	const;
	unsigned degree() 							const;
	Shape getShape() 							const;
	//---- Accessors for internal data storage types ----
	unsigned outerLD() 							const;
	unsigned outerRank() 						const;
	bool needUpdate()							const;
	bool owns_ranks() 							const;
	bool owns_tensor() 							const;
	bool owns_LD() 								const;
	bool isSubTensor() 							const;


	//--------------------------------------------------------Simple Mutators--------------------------------------------------------//
	void fill(const Scalar<number_type, TensorOperations>& s);
	void randomize(number_type lower_bound,number_type upper_bound);

	Tensor<number_type, TensorOperations>& reshape(Shape new_shape);
	Tensor<number_type, TensorOperations>& flatten();

	//--------------------------------------------------------Movement Semantics--------------------------------------------------------//

	void reset();
	void reset_post_move();

	//--------------------------------------------------------Debugging Tools--------------------------------------------------------//
	void print() const;
	void printDimensions() const;

	//--------------------------------------------------------Transposition--------------------------------------------------------//
	Tensor<number_type, TensorOperations> T() const;
	Tensor<number_type, TensorOperations> transpose() const;

	//--------------------------------------------------------Data Acessors--------------------------------------------------------//
	//----------Access a Scalar----------//

	number_type& operator()(unsigned index);
	  const number_type& operator()(unsigned index) const;

	//----------Access Dimension at index----------//
		  Tensor<number_type, TensorOperations>& operator[](unsigned index);
	const Tensor<number_type, TensorOperations>& operator[](unsigned index) const;

	//----------Access/Generate a SubTensor----------//

	Tensor<number_type, TensorOperations>& operator()(Shape index, Shape shape);
	const Tensor<number_type, TensorOperations>& operator()(Shape index, Shape shape) const;

	//----------clear chached tensors----------//
	void clearTensorCache();

public:
	//--------------------------------------------------------Assignment Operators--------------------------------------------------------//
	virtual Tensor<number_type, TensorOperations>& operator=(const Tensor<number_type, TensorOperations>& t);
	virtual Tensor<number_type, TensorOperations>& operator=(Tensor<number_type, TensorOperations>&& t);
	virtual Tensor<number_type, TensorOperations>& operator=(std::initializer_list<number_type> vector);
	virtual Tensor<number_type, TensorOperations>& operator=(const Scalar<number_type, TensorOperations>& s);
	virtual Tensor<number_type, TensorOperations>& operator=(number_type scalar);

	//--------------------------------------------------------Mathematics--------------------------------------------------------//
	//Mathematics ----- advanced //
	Tensor<number_type, TensorOperations> operator*  (const Tensor<number_type, TensorOperations>& t) const;
	Scalar<number_type, TensorOperations> corr(const Tensor<number_type, TensorOperations>& t) const;
	Tensor<number_type, TensorOperations> x_corr(unsigned move_dimensions, const Tensor<number_type, TensorOperations>& t) const;
	Tensor<number_type, TensorOperations> x_corr_FilterError(unsigned move_dimensions, const Tensor<number_type, TensorOperations>& t) const; //accepts the error of the output and returns the error of the filter
	Tensor<number_type, TensorOperations> x_corr_SignalError(unsigned move_dimensions, const Tensor<number_type, TensorOperations>& t) const; //accepts the error of the output and returns the error of the filter

	Tensor<number_type, TensorOperations> x_corr_full(unsigned move_dimensions, const Tensor<number_type, TensorOperations>& t) const;
	Tensor<number_type, TensorOperations> x_corr_stack(unsigned move_dimensions, const Tensor<number_type, TensorOperations>& t) const;
	Tensor<number_type, TensorOperations> x_corr_full_stack(unsigned move_dimensions, const Tensor<number_type, TensorOperations>& t) const;

	number_type max() const;
	number_type min() const;

	std::pair<number_type, Tensor<unsigned, TensorOperations>> max_index() const;
	std::pair<number_type, Tensor<unsigned, TensorOperations>> min_index() const;


//	Scalar<number_type, TensorOperations> conv(const Tensor<number_type, TensorOperations>& t) const;
//	Tensor<number_type, TensorOperations> x_conv(const Tensor<number_type, TensorOperations>& t) const;

	//Mathematics ----- pointwise //
	Tensor<number_type, TensorOperations> operator^(const Tensor<number_type, TensorOperations>& t) const;
	Tensor<number_type, TensorOperations> operator/(const Tensor<number_type, TensorOperations>& t) const;
	Tensor<number_type, TensorOperations> operator+(const Tensor<number_type, TensorOperations>& t) const;
	Tensor<number_type, TensorOperations> operator-(const Tensor<number_type, TensorOperations>& t) const;
	Tensor<number_type, TensorOperations> operator&(const Tensor<number_type, TensorOperations>& t) const;

	Tensor<number_type, TensorOperations>& operator^=(const Tensor<number_type, TensorOperations>& t);
	Tensor<number_type, TensorOperations>& operator/=(const Tensor<number_type, TensorOperations>& t);
	Tensor<number_type, TensorOperations>& operator+=(const Tensor<number_type, TensorOperations>& t);
	Tensor<number_type, TensorOperations>& operator-=(const Tensor<number_type, TensorOperations>& t);
	Tensor<number_type, TensorOperations>& operator&=(const Tensor<number_type, TensorOperations>& t);

	//--------------------------------------------------------Mathematics (Scalar)--------------------------------------------------------//

	Tensor<number_type, TensorOperations> operator^(const Scalar<number_type, TensorOperations>& t) const;
	Tensor<number_type, TensorOperations> operator/(const Scalar<number_type, TensorOperations>& t) const;
	Tensor<number_type, TensorOperations> operator+(const Scalar<number_type, TensorOperations>& t) const;
	Tensor<number_type, TensorOperations> operator-(const Scalar<number_type, TensorOperations>& t) const;
	Tensor<number_type, TensorOperations> operator&(const Scalar<number_type, TensorOperations>& t) const;

	Tensor<number_type, TensorOperations>& operator^=(const Scalar<number_type, TensorOperations>& t);
	Tensor<number_type, TensorOperations>& operator/=(const Scalar<number_type, TensorOperations>& t);
	Tensor<number_type, TensorOperations>& operator+=(const Scalar<number_type, TensorOperations>& t);
	Tensor<number_type, TensorOperations>& operator-=(const Scalar<number_type, TensorOperations>& t);
	Tensor<number_type, TensorOperations>& operator&=(const Scalar<number_type, TensorOperations>& t);

	Tensor<number_type, TensorOperations> operator^( number_type scal) const;
	Tensor<number_type, TensorOperations> operator/( number_type scal) const;
	Tensor<number_type, TensorOperations> operator+( number_type scal) const;
	Tensor<number_type, TensorOperations> operator-( number_type scal) const;
	Tensor<number_type, TensorOperations> operator&( number_type scal) const;

	Tensor<number_type, TensorOperations>& operator^=( number_type scal);
	Tensor<number_type, TensorOperations>& operator/=( number_type scal);
	Tensor<number_type, TensorOperations>& operator+=( number_type scal);
	Tensor<number_type, TensorOperations>& operator-=( number_type scal);
	Tensor<number_type, TensorOperations>& operator&=( number_type scal);


	//--------------------------------------------------------Boundry Checking--------------------------------------------------------//
	bool same_dimensions(const Tensor<number_type, TensorOperations>& t) const;
	void assert_same_dimensions(const Tensor<number_type, TensorOperations>& t) const;

	bool valid_dotProduct(const Tensor<number_type, TensorOperations>& t) const;
	void assert_dotProduct_dimensions(const Tensor<number_type, TensorOperations>& t) const;

	bool same_size(const Tensor<number_type, TensorOperations>& t) const;
	void assert_same_size(const Tensor<number_type, TensorOperations>& t) const;

	bool valid_convolution_target(const Tensor<number_type, TensorOperations>& t) const;
	bool valid_correlation_target(const Tensor<number_type, TensorOperations>& t) const;

	void assert_valid_convolution_target(const Tensor<number_type, TensorOperations>& t) const;
	void assert_valid_correlation_target(const Tensor<number_type, TensorOperations>& t) const;

	static void assert_isVector(const Tensor<number_type, TensorOperations>& t);
	static void assert_isMatrix(const Tensor<number_type, TensorOperations>& t);
	//--------------------------------------------------------Get Internal Data--------------------------------------------------------//
	void read(std::ifstream& is);
	void readCSV(std::ifstream& is);
	void readCSV(std::ifstream& is, unsigned numb_values);
	void write(std::ofstream& os);


	const number_type* data() const { return tensor;}
	number_type*& data() { return tensor; }
};
#endif
