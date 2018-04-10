#ifndef BlackCat_Tensor_super_h
#define BlackCat_Tensor_super_h

#include <string>
#include <iostream>
#include <initializer_list>
#include <unordered_map>
#include <utility>

#include "LinearAlgebraRoutines.h"
#include "CPU_Operations.h"
#include <vector>
#include "Scalar.h"
//ranks[0] = number of columns
//ranks[1] = number of rows
//ranks[2] = number of pages 
//ranks[3] ... etc

template<typename number_type>
class Tensor {
	typedef std::vector<unsigned> Shape;

private:

	const bool ownership;
	number_type* tensor = nullptr;
	unsigned* ranks = nullptr;
	unsigned order = 0;
	unsigned sz = 0;

	mutable std::unordered_map<unsigned, Tensor<number_type>*> IndexTensorMap;

//Private Constructors ----------------
	//dotproduct dimension constructor
	Tensor<number_type>(const Tensor<number_type>& m1, const Tensor<number_type>& m2); //dot product constructor
	//indexTensor constructor
	Tensor<number_type>(const Tensor<number_type>& super_tensor, unsigned index);
	//copy dim no trans
	Tensor<number_type>(const Tensor<number_type>& cpy_tensor, bool copy_values);
	//copy dimensions with transposed
	Tensor<number_type>(const Tensor<number_type>& cpy_tensor, bool copy_values,bool transposed);
protected:
	Tensor<number_type>* generateSub(unsigned index) const; //Subclass constructor should call (const Tensor<number_type>&, unsigned index) as constructor
public:
	//Empty Constructor
	Tensor<number_type>();
	//Copy Constructors
	Tensor<number_type>(Tensor<number_type> && cpy);
	Tensor<number_type>(const Tensor<number_type>& cpy);
	Tensor<number_type>(const Shape& shape);
	Shape getShape() const;
	//Other type of initializers
	Tensor<number_type>(std::initializer_list<unsigned> ranks);
	Tensor<number_type>(unsigned m, unsigned n, unsigned k, unsigned p);
	Tensor<number_type>(unsigned m, unsigned n, unsigned k);
	Tensor<number_type>(unsigned m, unsigned n);
	explicit Tensor<number_type>(unsigned m);
	virtual ~Tensor<number_type>();

	//Basic Accessors
	unsigned totalMatrices()			const;
	bool isInitialized() 				const;
	unsigned matrix_size() 				const;
	unsigned size() 					const;
	virtual unsigned rows() 			const;
	virtual unsigned cols() 			const;
	unsigned pages()					const;
	bool isMatrix() 					const;
	bool isSquare() 					const { return rows() == cols(); }
	bool isVector()						const;
	bool isScalar() 					const;
	unsigned rank(unsigned rank_index) 	const;
	unsigned degree() 					const;

	//Mutators
	void fill(number_type val);
	void fill(const Scalar<number_type>& s);
	void randomize(number_type lower_bound,number_type upper_bound);

	virtual Tensor<number_type>& reshape(std::initializer_list<unsigned> new_shape);
	virtual Tensor<number_type>& flatten();
	void reset();
	void reset_post_move();

	//Debugging
	void print() const;
	void printDimensions() const;

	//Transpose Methods
	const Tensor<number_type> T() const { return transpose(); }
	Tensor<number_type> transpose() const;

	//Data Accessors
	number_type& operator()(unsigned index);
	const number_type& operator()(unsigned index) const;

	Tensor<number_type>& operator[](unsigned index);
	const Tensor<number_type>& operator[](unsigned index) const;

	void clearSubTensorCache() const;

public:
	//Assignment operators
	virtual Tensor<number_type>& operator=(const Tensor<number_type>& t);
	virtual Tensor<number_type>& operator=(Tensor<number_type>&& t);
	virtual Tensor<number_type>& operator=(std::initializer_list<number_type> vector);
	virtual Tensor<number_type>& operator=(const Scalar<number_type>& s);

	//Mathematics operations
	Tensor<number_type> operator*(const Tensor<number_type>& t) const; //dot product of two matrices
	Tensor<number_type> operator->*(const Tensor<number_type>& t) const;
	Tensor<number_type> convolution(const Tensor<number_type>& t) const;
	Tensor<number_type> correlation(const Tensor<number_type>& t) const;

	Tensor<number_type> operator^(const Tensor<number_type>& t) const; //pointwise power of
	Tensor<number_type> operator/(const Tensor<number_type>& t) const; //pointwise division
	Tensor<number_type> operator+(const Tensor<number_type>& t) const; //pointwise addition
	Tensor<number_type> operator-(const Tensor<number_type>& t) const; //pointwise subtraction
	Tensor<number_type> operator&(const Tensor<number_type>& m) const; //pointise multiplication

	Tensor<number_type>& operator^=(const Tensor<number_type>& t); //pointwise power of and set
	Tensor<number_type>& operator/=(const Tensor<number_type>& t); //pointwise division and set
	Tensor<number_type>& operator+=(const Tensor<number_type>& t); //pointwise addition and set
	Tensor<number_type>& operator-=(const Tensor<number_type>& t); //pointwise subtraction and set
	Tensor<number_type>& operator&=(const Tensor<number_type>& m); //pointwise multiplication and set
	//Mathematics operators (By scalar)
	Tensor<number_type> operator^(const Scalar<number_type>& t) const; //power by scaler
	Tensor<number_type> operator/(const Scalar<number_type>& t) const; //division by scalar
	Tensor<number_type> operator+(const Scalar<number_type>& t) const; //addition by scalar
	Tensor<number_type> operator-(const Scalar<number_type>& t) const; //subtraction by scalar
	Tensor<number_type> operator&(const Scalar<number_type>& t) const; //multiplication by scalar

	Tensor<number_type>& operator^=(const Scalar<number_type>& t); //power by scalar and set
	Tensor<number_type>& operator/=(const Scalar<number_type>& t); //division by scalar and set
	Tensor<number_type>& operator+=(const Scalar<number_type>& t); //addition by scalar and set
	Tensor<number_type>& operator-=(const Scalar<number_type>& t); //subtraction by scalar and set
	Tensor<number_type>& operator&=(const Scalar<number_type>& t); //multiplication by scalar and set

	//Debugging / Bounds checking
	virtual bool same_dimensions(const Tensor<number_type>& t) const;
	void assert_same_dimensions(const Tensor<number_type>& t) const;

	virtual bool dotProduct_dimensions(const Tensor<number_type>& t) const;
	void assert_dotProduct_dimensions(const Tensor<number_type>& t) const;

	virtual bool same_size(const Tensor<number_type>& t) const;
	void assert_same_size(const Tensor<number_type>& t) const;

	virtual bool valid_convolution_target(const Tensor<number_type>& t) const;
	virtual bool valid_correlation_target(const Tensor<number_type>& t) const;

	void assert_valid_convolution_target(const Tensor<number_type>& t) const;
	void assert_valid_correlation_target(const Tensor<number_type>& t) const;

	static void assert_isVector(const Tensor<number_type>& t);
	static void assert_isMatrix(const Tensor<number_type>& t) { if (!t.isMatrix()) throw std::invalid_argument("assert isMatrix FAIL"); };
	const number_type* data() const { return tensor;}
	number_type*& data() { return tensor; }
};
#endif
