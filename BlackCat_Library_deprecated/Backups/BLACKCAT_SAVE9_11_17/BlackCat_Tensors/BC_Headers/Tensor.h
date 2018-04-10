#ifndef BlackCat_Tensor_super_h
#define BlackCat_Tensor_super_h

#include "LinearAlgebraRoutines.h"
#include "CPU_Operations.h"
#include "Scalar.h"

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

template<typename number_type>
class Tensor {

	//--------------------------------------------------------Members --------------------------------------------------------//
	const bool tensor_ownership;																				//Ownership - determines if Tensor controls life span of data
	const bool rank_ownership;
	const bool ld_ownership;
	const bool subTensor;

	//mutable Tensor<number_type>* self_transposed = nullptr;
	mutable bool needsUpdate = true;
	const Tensor<number_type>* parent = nullptr;

	number_type* tensor = nullptr;																		//tensor - the actual internal array of data
	unsigned* ranks = nullptr;																			//ranks -(dimensions) an array that stores the dimensions of the Tensor
	unsigned* ld = nullptr;																				//LeadingDimension - stores data regarding the leading_dimension

	unsigned sz = 0;																					//sz - total size of data [in non subTensors same as leading_dim]
	unsigned order = 0;																					//order - the number of degree in the matrix [dimensionality]

	mutable std::unordered_map<unsigned, Tensor<number_type>*> IndexTensorMap;							//index Tensor map - caches index tensors [for faster access]
	mutable std::unordered_map<std::pair<Shape, Shape>,Tensor<number_type>*,BC::indexShape_pairHasher, BC::indexShape_pairEqual> SubTensorMap;	//index Tensor map - caches index tensors [for faster access]

protected:
	//--------------------------------------------------------Private Constructors--------------------------------------------------------//

	Tensor<number_type>* generateIndexTensor(unsigned index) const;															//Sub_Tensor generator
	Tensor<number_type>* generateSubTensor(Shape index, Shape shape) const;

	Tensor<number_type>(const Tensor<number_type>& m1, const Tensor<number_type>& m2); 								//dot product constructor

	Tensor<number_type>(const Tensor<number_type>& super_tensor, unsigned index);									//Sub_Tensor constructor
	Tensor<number_type>(const Tensor<number_type>& super_tensor, Shape index, Shape shape);

	Tensor<number_type>(const Tensor<number_type>& cpy_tensor, bool copy_values);									//copy shape constructor
	Tensor<number_type>(const Tensor<number_type>& cpy_tensor, bool copy_values,bool transposed);					//copy shape transpose constructor


public:
	void alertUpdate() const {
		needsUpdate = true;

		if (parent)
			parent->alertUpdate();
	}

	//--------------------------------------------------------Public Constructors--------------------------------------------------------//
	Tensor<number_type>();																							//Empty initalize
	Tensor<number_type>(Tensor<number_type> && cpy);																//move constructor
	Tensor<number_type>(const Tensor<number_type>& cpy);															//copy constructor
	Tensor<number_type>(const Shape& shape);																		//Shape constructor
	//Other type of initializers
	Tensor<number_type>(std::initializer_list<unsigned> ranks);														//scaleable shape constructor
	Tensor<number_type>(unsigned m, unsigned n, unsigned k, unsigned p);
	Tensor<number_type>(unsigned m, unsigned n, unsigned k);
	Tensor<number_type>(unsigned m, unsigned n);

	explicit Tensor<number_type>(unsigned m);
	virtual ~Tensor<number_type>();

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
	unsigned outerLD() 							const { return ld[order - 1]; }
	unsigned outerRank() 						const { return ranks[order - 1]; }
	bool needUpdate()							const { return needsUpdate;}
	bool owns_ranks() 							const { return rank_ownership; }
	bool owns_tens0r() 							const { return tensor_ownership; }
	bool owns_LD() 								const { return ld; }
	bool isSubTensor() 							const { return subTensor; }


	//--------------------------------------------------------Simple Mutators--------------------------------------------------------//
	void fill(const Scalar<number_type>& s);
	void randomize(number_type lower_bound,number_type upper_bound);

	Tensor<number_type>& reshape(std::initializer_list<unsigned> new_shape);
	Tensor<number_type>& flatten();

	//--------------------------------------------------------Movement Semantics--------------------------------------------------------//

	void reset();
	void reset_post_move();

	//--------------------------------------------------------Debugging Tools--------------------------------------------------------//
	void print(const  number_type* tens, const unsigned* dims, const unsigned* dims_lead, unsigned index) const;
	void print() const;
	void printDimensions() const;

	//--------------------------------------------------------Transposition--------------------------------------------------------//
	//const Tensor<number_type>& T() const;
	Tensor<number_type> T() const;
	Tensor<number_type> transpose() const;

	//--------------------------------------------------------Data Acessors--------------------------------------------------------//
	//----------Access a Scalar----------//

	number_type& operator()(unsigned index);
	  const number_type& operator()(unsigned index) const;

	//----------Access Dimension at index----------//
		  Tensor<number_type>& operator[](unsigned index);
	const Tensor<number_type>& operator[](unsigned index) const;

	//----------Access/Generate a SubTensor----------//

	Tensor<number_type>& operator()(Shape index, Shape shape);
	const Tensor<number_type>& operator()(Shape index, Shape shape) const;

	//----------clear chached tensors----------//
	void clearTensorCache();

public:
	//--------------------------------------------------------Assignment Operators--------------------------------------------------------//
	virtual Tensor<number_type>& operator=(const Tensor<number_type>& t);
	virtual Tensor<number_type>& operator=(Tensor<number_type>&& t);
	virtual Tensor<number_type>& operator=(std::initializer_list<number_type> vector);
	virtual Tensor<number_type>& operator=(const Scalar<number_type>& s);

	//--------------------------------------------------------Mathematics--------------------------------------------------------//
	//Mathematics ----- advanced //
	Tensor<number_type> operator*  (const Tensor<number_type>& t) const;
	Scalar<number_type> corr(const Tensor<number_type>& t) const;
	Tensor<number_type> x_corr(unsigned corrdimensions, const Tensor<number_type>& t) const;

	Scalar<number_type> conv(const Tensor<number_type>& t) const;
	Tensor<number_type> x_conv(const Tensor<number_type>& t) const;

	//Tensor<number_type> corr2_noPad(const Tensor<number_type>& t) const;
	//Mathematics ----- pointwise //
	Tensor<number_type> operator^(const Tensor<number_type>& t) const;
	Tensor<number_type> operator/(const Tensor<number_type>& t) const;
	Tensor<number_type> operator+(const Tensor<number_type>& t) const;
	Tensor<number_type> operator-(const Tensor<number_type>& t) const;
	Tensor<number_type> operator&(const Tensor<number_type>& t) const;

	Tensor<number_type>& operator^=(const Tensor<number_type>& t);
	Tensor<number_type>& operator/=(const Tensor<number_type>& t);
	Tensor<number_type>& operator+=(const Tensor<number_type>& t);
	Tensor<number_type>& operator-=(const Tensor<number_type>& t);
	Tensor<number_type>& operator&=(const Tensor<number_type>& t);

	//--------------------------------------------------------Mathematics (Scalar)--------------------------------------------------------//

	Tensor<number_type> operator^(const Scalar<number_type>& t) const;
	Tensor<number_type> operator/(const Scalar<number_type>& t) const;
	Tensor<number_type> operator+(const Scalar<number_type>& t) const;
	Tensor<number_type> operator-(const Scalar<number_type>& t) const;
	Tensor<number_type> operator&(const Scalar<number_type>& t) const;

	Tensor<number_type>& operator^=(const Scalar<number_type>& t);
	Tensor<number_type>& operator/=(const Scalar<number_type>& t);
	Tensor<number_type>& operator+=(const Scalar<number_type>& t);
	Tensor<number_type>& operator-=(const Scalar<number_type>& t);
	Tensor<number_type>& operator&=(const Scalar<number_type>& t);

	//--------------------------------------------------------Boundry Checking--------------------------------------------------------//
	bool same_dimensions(const Tensor<number_type>& t) const;
	void assert_same_dimensions(const Tensor<number_type>& t) const;

	bool valid_dotProduct(const Tensor<number_type>& t) const;
	void assert_dotProduct_dimensions(const Tensor<number_type>& t) const;

	bool same_size(const Tensor<number_type>& t) const;
	void assert_same_size(const Tensor<number_type>& t) const;

	bool valid_convolution_target(const Tensor<number_type>& t) const;
	bool valid_correlation_target(const Tensor<number_type>& t) const;

	void assert_valid_convolution_target(const Tensor<number_type>& t) const;
	void assert_valid_correlation_target(const Tensor<number_type>& t) const;

	static void assert_isVector(const Tensor<number_type>& t);
	static void assert_isMatrix(const Tensor<number_type>& t);
	//--------------------------------------------------------Get Internal Data--------------------------------------------------------//
	void read(std::ifstream& is);
	void readCSV(std::ifstream& is);
	void readCSV(std::ifstream& is, unsigned numb_values);
	void write(std::ofstream& os);


	const number_type* data() const { return tensor;}
	number_type*& data() { return tensor; }
};
#endif
