/*
 * Dynamic_Shape_impl.h
 *
 *  Created on: Dec 21, 2017
 *      Author: joseph
 */

#ifndef DYNAMIC_SHAPE_IMPL_H_
#define DYNAMIC_SHAPE_IMPL_H_

namespace BC {


	template<class... params>
	struct init_inner;


	template<class front, class... params>
	struct init_inner<front, params...> {
		static int foo(int* ary, front f, params... p) {
			*ary = f;
			return f * init_inner<params...>::foo(++ary, p...);
		}
	};

	template<class front>
	struct init_inner<front> {
		static int foo(int* ary, front f) {
			*ary = f;
			return f;
		}
	};


	template<>
	struct init_inner<std::initializer_list<int>> {
		static int foo(int * ary, std::initializer_list<int> dims) {
			int sz = 1;
			for (unsigned i = 0; i < dims.size(); ++i) {
				sz *= dims.begin()[i];
				ary[i] = dims.begin()[i];
			}
				return sz;
		}
	};
	template<class... params>
	struct init_outer;


	template<class front, class... params>
	struct init_outer<front, params...> {
		static int foo(int* ary, front f, params... p) {
			*ary = f;
			return f * init_inner<params...>::bar(++ary, p...);
		}
		static int bar(int* ary, front f, params... p) {
			*ary = f * ary[-1];
			return f * init_inner<params...>::bar(++ary, p...);
		}
	};

	template<class front>
	struct init_outer<front> {
		static int foo(int* ary, front f) {
			*ary = f;
			return f;
		}
		static int bar(int* ary, front f) {
			*ary = f * ary[-1];
			return f;
		}
	};


	template<>
	struct init_outer<std::initializer_list<int>> {
		static int foo(int* ary, std::initializer_list<int> dims) {int sz = 1;
		ary[0] = dims.begin()[0];
		for (unsigned i = 1; i < dims.size(); ++i) {
			sz *= dims.begin()[i];
			ary[i] = dims.begin()[i] * ary[i -1];
		}
			return sz;
	}
	};



}

#endif /* DYNAMIC_SHAPE_IMPL_H_ */
