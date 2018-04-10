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

	template <typename t>
	friend class Tensor;

protected:
	//protected constructors
	bool ownership = true;
	number_type* scalar;
	//Index Tensor

public:
	Scalar<number_type>(const Scalar& s);
	Scalar<number_type>(Scalar&& s);
	Scalar<number_type>(number_type);
	Scalar<number_type>() {scalar = new number_type; }
	virtual Scalar<number_type>& operator = (const Scalar<number_type>& s);
	virtual Scalar<number_type>& operator = (number_type s);
	virtual Scalar<number_type>& operator = (Scalar<number_type>&& s);

	virtual ~Scalar<number_type>() {
		//if (ownership) Tensor_Operations<number_type>::destruction(scalar); }
	}
	//Access data
	virtual const number_type& operator () () const {return this->scalar[0];};
	virtual number_type& operator () () { return this->scalar[0];};

	//Mathematics operators (By scalar)
	Scalar<number_type> operator^(const Scalar<number_type>& t) const;
	Scalar<number_type> operator/(const Scalar<number_type>& t) const;
	Scalar<number_type> operator+(const Scalar<number_type>& t) const;
	Scalar<number_type> operator-(const Scalar<number_type>& t) const;
	Scalar<number_type> operator&(const Scalar<number_type>& t) const;

	virtual Scalar<number_type>& operator^=(const Scalar<number_type>& t);
	virtual Scalar<number_type>& operator/=(const Scalar<number_type>& t);
	virtual Scalar<number_type>& operator+=(const Scalar<number_type>& t);
	virtual Scalar<number_type>& operator-=(const Scalar<number_type>& t);
	virtual Scalar<number_type>& operator&=(const Scalar<number_type>& t);

	void print() const { std::cout << "[" << *scalar << "]"<< std::endl;};
	void printDimensions() const { std::cout << "[1]" << std::endl; };

	number_type* data() { return scalar; }
	const number_type* data() const { return scalar; }
};

template<typename number_type, typename ParentTensor>
class ID_Scalar : public Scalar<number_type> {



	const ParentTensor* parent;
public:

	ID_Scalar<number_type, ParentTensor>(const ParentTensor& mother, number_type* val) {
		this->ownership = false;
		this->scalar = val;
		parent = &mother;
	}
	~ID_Scalar<number_type,ParentTensor>() {
		parent = nullptr;
		this->scalar = nullptr;
		this->ownership = false;
	}

	number_type& operator () () { parent->alertUpdate(); return this->scalar[0];};
	Scalar<number_type>& operator = (const Scalar<number_type>& s) {
		parent->alertUpdate();
		return this->Scalar<number_type>::operator=(s);
	}


	Scalar<number_type>& operator = (number_type s)  {
		parent->alertUpdate();
		return this->Scalar<number_type>::operator=(s);
	}
	Scalar<number_type>& operator = (Scalar<number_type>&& s)  {
		parent->alertUpdate();
		return this->Scalar<number_type>::operator=(s);
	}
	Scalar<number_type>& operator^=(const Scalar<number_type>& t) {
		parent->alertUpdate();
		return this->operator^=(t);
	}
	Scalar<number_type>& operator/=(const Scalar<number_type>& t) {
		parent->alertUpdate();
		return this->operator/=(t);
	}
	Scalar<number_type>& operator+=(const Scalar<number_type>& t) {
		parent->alertUpdate();
		return this->operator+=(t);
	}
	Scalar<number_type>& operator-=(const Scalar<number_type>& t) {
		parent->alertUpdate();
		return this->operator-=(t);
	}
	Scalar<number_type>& operator&=(const Scalar<number_type>& t) {
		parent->alertUpdate();
		return this->operator&=(t);
	}
};


#endif /* SCALAR_H_ */
