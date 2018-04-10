/*
 * Scalar.h
 *
 *  Created on: Aug 15, 2017
 *      Author: joseph
 */

#ifndef SCALAR_H_
#define SCALAR_H_
#include "LinearAlgebraRoutines.h"
#include <iostream>
template <typename number_type>
class Scalar {

protected:
	template <typename t>
	friend class Tensor;

	number_type* scalar;
public:
	Scalar<number_type>(const Scalar& s);
	Scalar<number_type>(Scalar&& s);
	Scalar<number_type>(number_type);
	Scalar<number_type>() {scalar = nullptr; }
	virtual Scalar<number_type>& operator = (const Scalar<number_type>& s);
	virtual Scalar<number_type>& operator = (number_type s);
	virtual Scalar<number_type>& operator = (Scalar<number_type>&& s);

	virtual ~Scalar<number_type>() {if (scalar) Tensor_Operations<number_type>::destruction(scalar); }

	//Access data
	const number_type& operator () () const {return this->scalar[0];};
	number_type& operator () () { return this->scalar[0];};

	//Mathematics operators (By scalar)
	Scalar<number_type> operator^(const Scalar<number_type>& t) const;
	Scalar<number_type> operator/(const Scalar<number_type>& t) const;
	Scalar<number_type> operator+(const Scalar<number_type>& t) const;
	Scalar<number_type> operator-(const Scalar<number_type>& t) const;
	Scalar<number_type> operator&(const Scalar<number_type>& t) const;

	Scalar<number_type>& operator^=(const Scalar<number_type>& t);
	Scalar<number_type>& operator/=(const Scalar<number_type>& t);
	Scalar<number_type>& operator+=(const Scalar<number_type>& t);
	Scalar<number_type>& operator-=(const Scalar<number_type>& t);
	Scalar<number_type>& operator&=(const Scalar<number_type>& t);

	void print() const { std::cout << "[" << *scalar << "]"<< std::endl;};
	void printDimensions() const { std::cout << "[1]" << std::endl; };
};



#endif /* SCALAR_H_ */
