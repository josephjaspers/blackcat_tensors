/*
 * Dynamic_Shape.h
 *
 *  Created on: Jan 5, 2018
 *      Author: joseph
 */

#ifndef DYNAMIC_SHAPE_H_
#define DYNAMIC_SHAPE_H_

namespace BC {

template<class Math_lib>
struct Dynamic_Outer_Shape {

	static constexpr int COMPILE_TIME_LD_ROWS() { return 0; }
	static constexpr int COMPILE_TIME_LD_COLS() { return 0; }

	static constexpr bool DynamicOuterShape() { return true; }

	const int LD_RANK;
	const bool OWNERSHIP;
	int* outer_shape;

	struct OS_packet {
		OS_packet(int rank, int* dims) : order(rank), ranks(dims) {}

		int order;
		int* ranks;
	};
	Dynamic_Outer_Shape() : LD_RANK(0), OWNERSHIP(false), outer_shape(nullptr) { std::invalid_argument("Dynamic_Shape not init"); }
	OS_packet outer_packet() const {
		return OS_packet(LD_RANK, outer_shape);
	}

	using pass_type = OS_packet;


	Dynamic_Outer_Shape(int* dims, const int rank) : outer_shape(dims), LD_RANK(rank), OWNERSHIP(false) {}
	Dynamic_Outer_Shape(OS_packet packet) : outer_shape(packet.ranks), LD_RANK(packet.order), OWNERSHIP(false) {}

	Dynamic_Outer_Shape(std::initializer_list<int> dims) : LD_RANK(dims.size()), OWNERSHIP(true) {
		Math_lib::unified_initialize(outer_shape, LD_RANK);
		outer_shape[0] = dims.begin()[0];
		for (int i = 1; i < LD_RANK; ++i) {
			outer_shape[i] = dims.begin()[i] * outer_shape[i - 1];
		}
	}
	~Dynamic_Outer_Shape() {
		if (OWNERSHIP) {
			Math_lib::destroy(outer_shape);
		}
	}

	void printLeadingDimensions() const {
		for (int i = 0; i < LD_RANK; ++i) {
			std::cout << "[" << outer_shape[i] << "]";
		}
		std::cout << std::endl;
	}

	const int* getLD() const {
		return outer_shape;
	}
	int* getLD() {
		return outer_shape;
	}

	int LD_size()  const { return LD_RANK; }
	int LD_rows()  const { return outer_shape[0]; }
	int LD_cols()  const { return LD_RANK > 1 ? outer_shape[1] : 1; }
	int LD_depth() const { return LD_RANK > 2 ? outer_shape[2] : 1; }
	int LD_pages() const { return LD_RANK > 3 ? outer_shape[3] : 1; }
	int LD(int i)  const { return LD_RANK > i ? outer_shape[i] : 1; }

};

template<class Math_lib>
struct Dynamic_Inner_Shape {
	static constexpr int COMPILE_TIME_ROWS() { return 0; };
	static constexpr int COMPILE_TIME_COLS() { return 0; };

	static constexpr bool DynamicInnerShape() { return true; }
	const int RANK;

	const bool OWNERSHIP;
	int sz;
	int* inner_shape;

	struct IS_packet {
		IS_packet(int sz, int rank, int* dims) : size(sz), order(rank), ranks(dims) {}

		int order;
		int size;
		int* ranks;
	};

	using pass_type = IS_packet;


	IS_packet inner_packet() const {
		return IS_packet(sz,RANK, inner_shape);
	}
	IS_packet inner_sub_packet() const {
		return IS_packet(sz / dimension(RANK - 1), RANK - 1, inner_shape);
	}
	Dynamic_Inner_Shape() : RANK(0), OWNERSHIP(false), inner_shape(nullptr), sz(0) { std::invalid_argument("Dynamic_Shape not init"); }


	Dynamic_Inner_Shape(int* dims, const int rank) : inner_shape(dims), RANK(rank), OWNERSHIP(false) {
		sz = dims[rank];
	}
	Dynamic_Inner_Shape(IS_packet packet) : sz(packet.size), inner_shape(packet.ranks), RANK(packet.order), OWNERSHIP(false) {}


	Dynamic_Inner_Shape(std::initializer_list<int> dims) : RANK(dims.size()), OWNERSHIP(true) {
		Math_lib::unified_initialize(inner_shape, RANK);
		inner_shape[0] = dims.begin()[0];
		sz = inner_shape[0];
		for (int i = 1; i < RANK; ++i) {
			inner_shape[i] = dims.begin()[i];
			sz *= inner_shape[i];
		}
	}

	~Dynamic_Inner_Shape() {
		if (OWNERSHIP) {
			Math_lib::destroy(inner_shape);
		}
	}

	void printDimensions() const {
		for (int i = 0; i < RANK; ++i) {
			std::cout << "[" << inner_shape[i] << "]";
		}
		std::cout << std::endl;
	}
	const int* getShape() const {
		return inner_shape;
	}

	int* getShape() {
		return inner_shape;
	}

	int order() const { return RANK; }
	int size()  const { return sz; }
	int rows()  const { return inner_shape[0]; }
	int cols()  const { return RANK > 1 ? inner_shape[1] : 1; }
	int depth() const { return RANK > 2 ? inner_shape[2] : 1; }
	int pages() const { return RANK > 3 ? inner_shape[3] : 1; }

	int dimension(int i) const { return RANK > i ? inner_shape[i]: 1; }


};

}


#endif /* DYNAMIC_SHAPE_H_ */
