
#include "Vector.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename n>
unsigned Vector<n>::rows() const {
	return this->size();
}
template<typename n>
unsigned Vector<n>::cols() const {
	return 1;
}
