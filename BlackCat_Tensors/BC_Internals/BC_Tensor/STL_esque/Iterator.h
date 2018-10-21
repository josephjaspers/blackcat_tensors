/*
 * Iterator.h
 *
 *  Created on: Oct 18, 2018
 *      Author: joseph
 */

#ifndef ITERATOR_H_
#define ITERATOR_H_

namespace BC {
namespace module {
namespace stl {

template<int dimension, class tensor_t>
struct Iterator {

	using iterator_category = std::random_access_iterator_tag;
	using value_type = decltype(std::declval<tensor_t>().slice(0));
	using difference_type = int;
	using pointer = tensor_t&;
	using reference = value_type;

private:
	bool same_ptr(const Iterator& iter) const { return tensor.data() == iter.tensor.data(); }
	bool same_index(const Iterator& iter) const { return index == iter.index; }
public:

	tensor_t& tensor;
	mutable int index= 0;

	Iterator(tensor_t& tensor_, int index_=0)
	: tensor(tensor_), index(index_) {}

	bool operator == (const Iterator& iter) {
		return same_index(iter) && same_ptr(iter);
	}
	bool operator != (const Iterator& iter) {
		return !((*this)==iter);
	}
	bool operator > (const Iterator& iter) {
		return index > iter.index && same_ptr(iter);
	}
	bool operator < (const Iterator& iter) {
		return index < iter.index && same_ptr(iter);
	}
	bool operator >= (const Iterator& iter) {
		return index >= iter.index && same_ptr(iter);
	}
	bool operator <= (const Iterator& iter) {
		return index <= iter.index && same_ptr(iter);
	}

	Iterator& operator ++ ()    { ++index; return *this; }
	Iterator& operator ++ (int) { ++index; return *this; }
	Iterator& operator -- ()    { --index; return *this; }
	Iterator& operator -- (int) { --index; return *this; }
	Iterator& operator += (int dist)    { index += dist; return *this; }
	Iterator& operator -= (int dist)    { index -= dist; return *this; }
	Iterator& operator + (int dist)    { return Iterator(tensor, index + dist); }
	Iterator& operator - (int dist)    { return Iterator(tensor, index - dist); }

	reference operator*() const { return tensor[index]; }

	auto begin() {
		return forward_iterator_begin(tensor[index]);
	}
	auto cbegin() const {
		return forward_iteartor_begin(tensor[index]);
	}
	auto end() {
		return forward_iterator_end(tensor[index]);
	}
	auto cend() const {
		return forward_iteartor_end(tensor[index]);
	}
};

template<class derived_t>
auto forward_iterator_begin(derived_t& derived) {
	 return Iterator<derived_t::DIMS(), derived_t>(derived, 0);
}
template<class derived_t>
auto forward_iterator_end(derived_t& derived) {
	 return Iterator<derived_t::DIMS(), derived_t>(derived, derived.outer_dimension());
}



template<int dimension, class tensor_t>
struct Reverse_Iterator {

	using iterator_category = std::random_access_iterator_tag;
	using value_type = decltype(std::declval<tensor_t>().slice(0));
	using difference_type = int;
	using pointer = tensor_t&;
	using reference = value_type;

private:
	bool same_ptr(const Reverse_Iterator& iter) const { return tensor.data() == iter.tensor.data(); }
	bool same_index(const Reverse_Iterator& iter) const { return index == iter.index; }
public:

	tensor_t& tensor;
	mutable int index=tensor.outer_dimension()-1;

	Reverse_Iterator(tensor_t& tensor_, int index_)
	: tensor(tensor_), index(index_) {}
	Reverse_Iterator(tensor_t& tensor_)
	: tensor(tensor_) {}

	//Reverse iterator is identical to Forward_Iterator, but all the signs are reversed
	bool operator == (const Reverse_Iterator& iter) {
		return same_index(iter) && same_ptr(iter);
	}
	bool operator != (const Reverse_Iterator& iter) {
		return !((*this)==iter);
	}
	bool operator > (const Reverse_Iterator& iter) {
		return index < iter.index && same_ptr(iter);
	}
	bool operator < (const Reverse_Iterator& iter) {
		return index > iter.index && same_ptr(iter);
	}
	bool operator >= (const Reverse_Iterator& iter) {
		return index <= iter.index && same_ptr(iter);
	}
	bool operator <= (const Reverse_Iterator& iter) {
		return index >= iter.index && same_ptr(iter);
	}

	Reverse_Iterator& operator ++ ()    { --index; return *this; }
	Reverse_Iterator& operator ++ (int) { --index; return *this; }
	Reverse_Iterator& operator -- ()    { ++index; return *this; }
	Reverse_Iterator& operator -- (int) { ++index; return *this; }
	Reverse_Iterator& operator += (int dist)    { index -= dist; return *this; }
	Reverse_Iterator& operator -= (int dist)    { index += dist; return *this; }
	Reverse_Iterator& operator + (int dist)    { return Reverse_Iterator(tensor, index - dist); }
	Reverse_Iterator& operator - (int dist)    { return Reverse_Iterator(tensor, index + dist); }

	auto operator*() const { return tensor[index]; }
	auto operator*() { return tensor[index]; }

	auto begin() {
		return reverse_iterator_begin(tensor[index]);
	}
	auto cbegin() const {
		return reverse_iteartor_begin(tensor[index]);
	}
	auto end() {
		return reverse_iterator_end(tensor[index]);
	}
	auto cend() const {
		return reverse_iteartor_end(tensor[index]);
	}

};

template<class derived_t>
auto reverse_iterator_begin(derived_t& derived) {
	 return Reverse_Iterator<derived_t::DIMS(), derived_t>(derived, derived.outer_dimension()-1);
}

template<class derived_t>
auto reverse_iterator_end(derived_t& derived) {
	 return Reverse_Iterator<derived_t::DIMS(), derived_t>(derived, -1);
}

}
}
}


#endif /* ITERATOR_H_ */
