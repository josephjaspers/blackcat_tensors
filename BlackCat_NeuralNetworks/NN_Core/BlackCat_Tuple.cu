
#ifndef BLACKCAT_TUPLE
#define BLACKCAT_TUPLE

namespace BC {


//Tuple Array --> recursively inherits from itself to create a tuple
//Tuple arrays have a single "get" method which returns data and a using decleration of its base class
template<int N, class Data, class... set>
struct tuple_array : tuple_array<N+1, set...> {
	using parent = tuple_array<N+1, set...>;
	static constexpr int value = N;
	Data data;
	Data& get() { return data; }
	const Data& get() const { return data; }

	tuple_array<N, Data, set...>(Data dat, set... s) : tuple_array<N+1, set...>(s...), data(dat) {}
};

//Tuple array base case
template<int N, class Data>
struct tuple_array<N, Data> {
	static constexpr int value = N;
	Data data;
	Data& get() { return data; }
	const Data& get() const { return data; }

	tuple_array(Data dat) : data(dat) {}
};

//Caster recursively cast the tuple<int N, class... set> to its parent class 'i' times (to enable using get).
template<int i, int N> struct caster {
	template<class... set>
	auto& foo(tuple_array<N, set...> ta) {
		return caster<i + 1, N>::foo(static_cast<typename tuple_array<N, set...>::parent>(ta));
	}
	template<class... set>
	const auto& foo(const tuple_array<N, set...> ta) {
		return caster<i + 1, N>::foo(static_cast<typename tuple_array<N, set...>::parent>(ta));
	}
};

//Just an implementation detail, replaces the integer of the tuple with the correct index
//This detail enables classes to inherit from the same type multiple times.
template<int ,class> struct sub_i;
template<int X, template<int i, class...> class tuple, int i, class... set>
struct sub_i<X, tuple<i, set...>> {
	using type = tuple<X, set...>;
};

//GETTER returns the correct cast_type
template<int i, class f, class... set>
struct GETTER {
	using data = typename GETTER<i - 1, set...>::type;
	static constexpr int index = i;
	using cast_type = typename GETTER<i - 1, set...>::cast_type;
	using type = typename sub_i<i, cast_type>::type;

};
template<class f, class... set>
struct GETTER<0, f, set...> {
	using data = f;
	using cast_type = tuple_array<0, f, set...>;
	using type = typename sub_i<0, cast_type>::type;
};



template<class ... set>
class Tuple {
public:
	tuple_array<0, set...> array;
	Tuple(set... s) : array(s...) {}
	template<int i> auto& get() {
		return static_cast<typename GETTER<i, set...>::type&>(array).get();
	}
	template<int i> const auto& get() const {
		return static_cast<const typename GETTER<i, set...>::type&>(array).get();
	}
};

template<int N, class ... set>
auto& get(Tuple<set...>& t) {
	return static_cast<typename GETTER<N, set...>::type&>(t.array).get();
}
template<int N, class ... set>
const auto& get(const Tuple<set...>& t) {
	return static_cast<const typename GETTER<N, set...>::type&>(t.array).get();
}
}
#endif
