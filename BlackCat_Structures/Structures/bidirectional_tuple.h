/*
 * bidirectional_tuple.h
 *
 *  Created on: Apr 28, 2018
 *      Author: joseph
 */

#ifndef BIDIRECTIONAL_TUPLE_H_
#define BIDIRECTIONAL_TUPLE_H_

namespace BC {
namespace Structure {
struct  HEAD;
template<class derived, class...> struct bidirectional_tuple;
/*
 * Abstract
 *
 */


template<class... Ts>
struct Tuple {

	bidirectional_tuple<HEAD, Ts...> data;

	template<class... Us>
	Tuple(Us... data_) : data(data_...) {}


	template<int x> const auto& get() const { return get_impl<x>(data.head()); }
	template<int x> 	  auto& get() 		{ return get_impl<x>(data.head()); }

	auto& head() { return data.head(); }
	auto& tail() { return data.tail(); }
	auto& next() { return head().next(); }

	const auto& head() const { return data.head(); }
	const auto& tail() const { return data.tail(); }
	const auto& next() const { return head().next(); }
	bool prev() const { return false; }


	template<class functor>
	void for_each(functor f) {
		for_each_impl(data.head(), f);
	}


	template<int x, class Node>
	auto& get_impl(Node& n) {
		if constexpr (x == 0)
				return n.data();
		else
			return get_impl<x-1>(n.next());
	}
	template<int x, class Node>
	const auto& get_impl(Node& n) const {
		if constexpr (x == 0)
				return n.data();
		else
			return get_impl<x-1>(n.next());
	}

	template<class functor, class Node>
	void for_each_impl(Node& n, functor f) {
		f(n.data());

		if constexpr (n.hasNext())
			for_each_impl(n.next(), f);
	}

};







	//TAIL
	template<class derived, class T>
	struct bidirectional_tuple<derived,  T> {

		using prev_ = derived;
		using type = T;

		T data_;



		template<class... params>
		bidirectional_tuple(params... x) : data_(x...) {}

		constexpr bool hasNext() const { return false; }
		constexpr bool hasPrev() const { return true; }

		const auto& tail() const { return *this; }
			  auto& tail() 		 { return *this; }

		const auto& head() const { return prev().head(); }
			  auto& head()  	 { return prev().head(); }

		const bool next() const { return false; }
			  bool next() 		{ return false; }

		const auto& prev() const { return static_cast<prev_&>(*this).data(); }
			  auto& prev() 		 { return static_cast<prev_&>(*this).data(); }

		const T& data() const { return data_; }
			  T& data()  	 { return  data_; }
	};

	//BODY
	template<class derived, class T, class... Ts>
	struct bidirectional_tuple<derived, T, Ts...> : bidirectional_tuple<bidirectional_tuple<derived, T, Ts...>, Ts...> {

		using prev_ = derived;
		using next_ = bidirectional_tuple<bidirectional_tuple<derived, T, Ts...>, Ts...>;
		using type = T;

		T data_;

		constexpr bool hasNext() const { return true; }
		constexpr bool hasPrev() const { return true; }

		template<class param, class... intiailizers> bidirectional_tuple(param x, intiailizers... params)
				: data_(x), next_(params...) {}
		bidirectional_tuple() = default;


		const auto& tail() const { return next().tail(); }
		const auto& head() const { return prev().head(); }

		auto& tail() { return next().tail(); }
		auto& head() { return prev().head(); }

			  auto& prev()  	 { return static_cast<prev_&>(*this); }
		const auto& prev() const { return static_cast<prev_&>(*this); }

			  auto& next()    	 { return static_cast<next_&>(*this); }
		const auto& next() const { return static_cast<next_&>(*this); }

		const auto& data() const { return data_; }
		 	  auto& data()  	 { return data_; }
	};

	//HEAD
	struct HEAD;

	template<class T, class... Ts>
	struct bidirectional_tuple<HEAD, T, Ts...>
		: public bidirectional_tuple<bidirectional_tuple<HEAD, T, Ts...>, Ts...> {

		using next_ = bidirectional_tuple<bidirectional_tuple<HEAD, T, Ts...>, Ts...>;
		using type = T;

		T data_;

		template<class param, class... intiailizers> bidirectional_tuple(param x, intiailizers... params)
				:next_(params...), data_(x) {}
		bidirectional_tuple() = default;


		constexpr bool hasNext() const { return true; }
		constexpr bool hasPrev() const { return false; }

		const auto& data() const { return data_; }
			  auto& data()  	 { return data_; }
		const auto& tail() const { return next().tail(); }
			  auto& tail() 		 { return next().tail(); }
		const auto& head() const { return *this; }
			  auto& head()  	 { return *this; }
		const auto& next() const { return static_cast<next_&>(*this); }
			  auto& next()  	 { return static_cast<next_&>(*this); }
		const bool prev() const { return false; }
			  bool prev()  	 	{ return false; }
	};

}
}




#endif /* BIDIRECTIONAL_TUPLE_H_ */
