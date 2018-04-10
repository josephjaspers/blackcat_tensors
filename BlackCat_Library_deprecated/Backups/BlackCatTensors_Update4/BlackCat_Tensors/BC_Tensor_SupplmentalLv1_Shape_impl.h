/*
 * BC_Tensor_Shape_impl.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_SUPPLMENTALLV1_SHAPE_IMPL_H_
#define BC_TENSOR_SUPPLMENTALLV1_SHAPE_IMPL_H_

/*
 * Template Metaprogramming -- handles the methods related to dimensionality
 */

namespace BC_Shape_Identity_impl {

	template<int a, int b>
	struct max {
		static constexpr int value = a > b ? a : b;
	};
	template<int a, int b>
	struct min {
		static constexpr int value = a < b ? a : b;
	};

	template<int curr_dim, int ... dimensions>
	struct size_helper {
		static constexpr int value = curr_dim * size_helper<dimensions...>::value;
	};
	template<int curr_dim>
	struct size_helper<curr_dim> {
		static constexpr int value = curr_dim;
	};

	template<int ...dimensions>
	constexpr int size() {
		return size_helper<dimensions...>::value;
	}

	template<int ... dims>
	struct first {
		static constexpr int value = 0;
	};
	template<int f, int ... dims>
	struct first<f, dims...> {
		static constexpr int value = f;
	};

	template<int index, int ... dims>
	struct dimension {
		static constexpr int value = 0;
	};

	template<int index, int curr_dim, int ... dims>
	struct dimension<index, curr_dim, dims...> {
		static constexpr int value = index == 1 ? curr_dim : dimension<max<index - 1, 0>::value, dims...>::value;
	};
	template<int index, int curr_dim>
	struct dimension<index, curr_dim> {
		static constexpr int value = index == 1 ? curr_dim : 1;
	};

	//get row from dimensions
	template<int r, int ... dimensions>
	constexpr int row() {
		return r;
	}

	//get col from dimensions
	template<int ... dimensions>
	constexpr int col() {
		return dimension<2, dimensions...>::value;
	}

	//get depth (3rd dim) from dimensions
	template<int ... dimensions>
	constexpr int depth() {
		return dimension<3, dimensions...>::value;
	}

	//get pages (4th dim) from dimensions
	template<int ... dimensions>
	constexpr int pages() {
		return dimension<4, dimensions...>::value;
	}

	template<int ... dimensions>
	constexpr int books() {
		return dimension<5, dimensions...>::value;
	}
	template<int ... dimensions>
	constexpr int libraries() {
		return dimension<6, dimensions...>::value;
	}
}
#endif /* BC_TENSOR_SUPPLMENTALLV1_SHAPE_IMPL_H_ */
