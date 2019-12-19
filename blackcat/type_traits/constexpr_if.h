/*
 * ConstexprIf.h
 *
 *  Created on: Jun 30, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_CONSTEXPRIF_H_
#define BLACKCAT_CONSTEXPRIF_H_

namespace BC {
namespace traits {


//-------------------------- constexpr ternary -----------------------//
template<bool>
struct constexpr_ternary_impl {
    template<class f1, class f2>
    static auto impl(f1 true_path, f2 false_path) {
        return true_path();
    }
};

template<>
struct constexpr_ternary_impl<false> {
    template<class f1, class f2>
    static auto impl(f1 true_path, f2 false_path) {
        return false_path();
    }
};

template<bool cond, class f1, class f2>
auto constexpr_ternary(f1 true_path, f2 false_path) {
    return constexpr_ternary_impl<cond>::impl(true_path, false_path);
}

//-------------------------- constexpr if -----------------------//
/*
 * C++ 11/14 version of constexpr if.  (Required because NVCC doesn't support C++17)
 *
 * Accepts a constexpr bool template argument, and one or two functors (that take no arguments)
 * if true, calls and returns the first functor, else the second.
 *
 * Example:
 *
 * int main() {
 *  static constexpr bool value = false;
 *	return BC::meta:constexpr_if<false>(
 *		[]() { std::cout << " constexpr_boolean is true " << std::endl;  return true;},
 *		[]() { std::cout << " constexpr_boolean is false " << std::endl; return false; }
 *	);
 *}
 * output:  constexpr_boolean is false
 */

template<bool>
struct constexpr_if_impl {
    template<class f1>
    static auto impl(f1 path) {
        return path();
    }
};
template<>
struct constexpr_if_impl<false> {
    template<class f1>
    static void impl(f1 path) {}
};
template<bool b,class f>
auto constexpr_if(f path) {
    return constexpr_if_impl<b>::impl(path);
}
template<bool cond, class f1, class f2>
auto constexpr_if(f1 true_path, f2 false_path) {
    return constexpr_ternary_impl<cond>::impl(true_path, false_path);
}

template<bool, class...> struct Constexpr_Else_If;

template<class f1, class f2>
struct Constexpr_Else_If<true, f1, f2> {
	mutable f1 f1_;
	f2 f2_;

	template<int ADL=0>
	auto operator () () {
		return f1_();
	}
	template<int ADL=0>
	auto operator () () const {
		return f1_();
	}
};
template<class f1, class f2>
struct Constexpr_Else_If<false, f1, f2> {
	f1 f1_;
	f2 f2_;

	template<int ADL=0>
	auto operator () () {
		return f2_();
	}
	template<int ADL=0>
	auto operator () () const {
		return f2_();
	}
};
template<class f1>
struct Constexpr_Else_If<true, f1> {
	f1 f1_;

	template<int ADL=0>
	auto operator () () {
		return f1_();
	}
	template<int ADL=0>
	auto operator () () const {
		return f1_();
	}
};
template<class f1>
struct Constexpr_Else_If<false, f1> {
	f1 f1_;

	template<int ADL=0>
	void operator () () {
	}
	template<int ADL=0>
	void operator () () const {
	}
};

template<bool cond, class f1>
auto constexpr_else_if(f1 f1_) {
	return Constexpr_Else_If<cond, f1>{f1_};
}
template<bool cond, class f1, class f2>
auto constexpr_else_if(f1 f1_, f2 f2_) {
	return Constexpr_Else_If<cond, f1, f2>{f1_, f2_};
}


template<class f1>
struct Constexpr_Else {
	f1 f1_;

	template<int adl=0>
	auto operator () () {
		return f1_();
	}
	template<int adl=0>
	auto operator () () const {
		return f1_();
	}
};

template<class f1>
Constexpr_Else<f1> constexpr_else(f1 f1_) {
	return Constexpr_Else<f1>{f1_};
}



}
}




#endif /* CONSTEXPRIF_H_ */
