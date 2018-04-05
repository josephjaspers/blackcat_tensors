/*
 * List.h
 *
 *  Created on: Apr 5, 2018
 *      Author: joseph
 */

#ifndef LIST_H_
#define LIST_H_

template<class... params>
struct data_list {
	int  data() {
		return 0;
	}

	bool isEmpty() {
		return true;
	}
	data_list& next() {
		return *this;
	}
	data_list& next() const {
		return *this;
	}
};

template<class first, class... params>
struct data_list<first, params...> : data_list<params...> {

	const first& data_;

	bool isEmpty() {
		return false;
	}

	const auto& data() {
		return data_;
	}

	data_list(const first& f, const params&... p) : data_(f), list<params...>(p...){}

	auto& next() {
		return static_cast<list<params...>&>(*this);
	}
	const auto& next() const {
		return static_cast<list<params...>&>(*this);
	}
};






#endif /* LIST_H_ */
