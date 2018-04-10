#include "Scalar.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"
#include <math.h>

	template<typename number_type, typename ScalarOperations> Scalar<number_type, ScalarOperations> Scalar<number_type, ScalarOperations>::operator^(const Scalar<number_type, ScalarOperations>& t) const {
		return (pow(*scalar, *t.scalar));
	}
	template<typename number_type, typename ScalarOperations> Scalar<number_type, ScalarOperations> Scalar<number_type, ScalarOperations>::operator/(const Scalar<number_type, ScalarOperations>& t) const {
		return *scalar / *t.scalar;
	}
	template<typename number_type, typename ScalarOperations> Scalar<number_type, ScalarOperations> Scalar<number_type, ScalarOperations>::operator+(const Scalar<number_type, ScalarOperations>& t) const {
		return *scalar + *t.scalar;

	}
	template<typename number_type, typename ScalarOperations> Scalar<number_type, ScalarOperations> Scalar<number_type, ScalarOperations>::operator-(const Scalar<number_type, ScalarOperations>& t) const {
		return *scalar - *t.scalar;

	}
	template<typename number_type, typename ScalarOperations> Scalar<number_type, ScalarOperations> Scalar<number_type, ScalarOperations>::operator&(const Scalar<number_type, ScalarOperations>& t) const {
		return *scalar * *t.scalar;

	}

	template<typename number_type, typename ScalarOperations> Scalar<number_type, ScalarOperations>& Scalar<number_type, ScalarOperations>::operator^=(const Scalar<number_type, ScalarOperations>& t) {
		 *scalar = pow(*scalar, *t.scalar);
		 return *this;
	}
	template<typename number_type, typename ScalarOperations> Scalar<number_type, ScalarOperations>& Scalar<number_type, ScalarOperations>::operator/=(const Scalar<number_type, ScalarOperations>& t) {
		 *scalar /= *t.scalar;
		 return *this;
	}
	template<typename number_type, typename ScalarOperations> Scalar<number_type, ScalarOperations>& Scalar<number_type, ScalarOperations>::operator+=(const Scalar<number_type, ScalarOperations>& t) {
		 *scalar += *t.scalar;
		 return *this;
	}
	template<typename number_type, typename ScalarOperations> Scalar<number_type, ScalarOperations>& Scalar<number_type, ScalarOperations>::operator-=(const Scalar<number_type, ScalarOperations>& t) {
		 *scalar -= *t.scalar;
		 return *this;
	}
	template<typename number_type, typename ScalarOperations> Scalar<number_type, ScalarOperations>& Scalar<number_type, ScalarOperations>::operator&=(const Scalar<number_type, ScalarOperations>& t) {
		 *scalar *= *t.scalar;
		 return *this;
	}
