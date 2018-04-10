#include "Scalar.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

	template<typename number_type> Scalar<number_type> Scalar<number_type>::operator^(const Scalar<number_type>& t) const {
		return (pow(*scalar, *t.scalar));
	}
	template<typename number_type> Scalar<number_type> Scalar<number_type>::operator/(const Scalar<number_type>& t) const {
		return *scalar / *t.scalar;
	}
	template<typename number_type> Scalar<number_type> Scalar<number_type>::operator+(const Scalar<number_type>& t) const {
		return *scalar + *t.scalar;

	}
	template<typename number_type> Scalar<number_type> Scalar<number_type>::operator-(const Scalar<number_type>& t) const {
		return *scalar - *t.scalar;

	}
	template<typename number_type> Scalar<number_type> Scalar<number_type>::operator&(const Scalar<number_type>& t) const {
		return *scalar * *t.scalar;

	}

	template<typename number_type> Scalar<number_type>& Scalar<number_type>::operator^=(const Scalar<number_type>& t) {
		 *scalar = pow(*scalar, *t.scalar);
		 return *this;
	}
	template<typename number_type> Scalar<number_type>& Scalar<number_type>::operator/=(const Scalar<number_type>& t) {
		 *scalar /= *t.scalar;
		 return *this;
	}
	template<typename number_type> Scalar<number_type>& Scalar<number_type>::operator+=(const Scalar<number_type>& t) {
		 *scalar += *t.scalar;
		 return *this;
	}
	template<typename number_type> Scalar<number_type>& Scalar<number_type>::operator-=(const Scalar<number_type>& t) {
		 *scalar -= *t.scalar;
		 return *this;
	}
	template<typename number_type> Scalar<number_type>& Scalar<number_type>::operator&=(const Scalar<number_type>& t) {
		 *scalar *= *t.scalar;
		 return *this;
	}
