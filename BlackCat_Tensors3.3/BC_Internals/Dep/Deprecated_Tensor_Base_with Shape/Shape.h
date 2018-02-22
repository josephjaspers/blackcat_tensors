///*
// * Shape.h
// *
// *  Created on: Jan 16, 2018
// *      Author: joseph
// */
//
//#ifndef SHAPE_H_
//#define SHAPE_H_
//
//namespace BC {
//
////default
//template<int COMPILE_TIME_ORDER, class shape = int[COMPILE_TIME_ORDER]>
//struct shape_packet {
//public:
//	void printDimensions() const { for (int i = 0; i < COMPILE_TIME_ORDER; ++i) { std::cout <<"[" << this->inner_shape[i] << "]"; } std::cout << "/n"; }
//
//	int sz = 0;
//	int size() const { return sz; }
//	int rows() const { return COMPILE_TIME_ORDER > 0 ? inner_shape[0] : 1;}
//	int cols() const { return COMPILE_TIME_ORDER > 1 ? inner_shape[1] : 1;}
//
//	shape inner_shape;
//	shape outer_shape;
//
//	const auto InnerShape() const { return inner_shape; }
//	const auto OuterShape() const { return outer_shape; }
//
//	shape_packet<COMPILE_TIME_ORDER - 1, int*> access_packet() const {
//		return shape_packet<COMPILE_TIME_ORDER - 1, int*>(sz / inner_shape[COMPILE_TIME_ORDER - 1], inner_shape, outer_shape);
//	}
//	shape_packet<COMPILE_TIME_ORDER, int*> expression_packet() const {
//		return shape_packet<COMPILE_TIME_ORDER, int*>(sz, inner_shape, outer_shape);
//	}
//	~shape_packet() = default;
//	shape_packet(const int* ary) {
//		if /*constexpr*/ (COMPILE_TIME_ORDER == 0) {
//			sz = 1;
//		}
//		if /*constexpr*/ (COMPILE_TIME_ORDER == 1) {
//			inner_shape[0] = *ary;
//			sz = *inner_shape;
//		} else if /*constexpr*/  (COMPILE_TIME_ORDER  == 2) {
//			inner_shape[0] = ary[0];
//			inner_shape[1] = ary[1];
//			sz = inner_shape[0] * inner_shape[1];
//		} else {
//			sz = 1;
//			for (int i = 0; i < COMPILE_TIME_ORDER; ++i) {
//				inner_shape[i] = ary[i];
//				sz *= inner_shape[i];
//			}
//		}
//	}
//};
//template<>
//struct shape_packet<1, int[1]> {
//public:
//	void printDimensions() const {  std::cout <<"[" << sz << "]"; std::cout << "/n"; }
//	static constexpr int one = 1;
//
//	int sz = 0;
//	int size() const { return sz; }
//	int rows() const { return sz; }
//	int cols() const { return 1;  }
//
//
//
//	const auto InnerShape() const { return &sz; }
//	const auto OuterShape() const { return &one; }
//
//
//	shape_packet<1, int[1]> expression_packet() const {
//		return shape_packet<1, int[1]>(sz);
//	}
//	shape_packet(const int* dim) {
//		sz = * dim;
//	}
//	shape_packet(int, const int* dim, const int* o_dim) {
//		sz = * dim;
//	}
//	shape_packet(shape_packet&& p) {
//		sz = p.sz;
//	}
//	shape_packet(const int dim) {
//		sz = dim;
//	}
//	~shape_packet() = default;
//};
//
//template<int COMPILE_TIME_ORDER>
//struct shape_packet<COMPILE_TIME_ORDER, int*> {
//public:
//	using shape = int*;
//	void printDimensions() const { for (int i = 0; i < COMPILE_TIME_ORDER; ++i) { std::cout <<"[" << this->inner_shape[i] << "]"; } std::cout << "/n"; }
//
//	int sz = 0;
//	int size() const { return sz; }
//	int rows() const { return COMPILE_TIME_ORDER > 0 ? inner_shape[0] : 1;}
//	int cols() const { return COMPILE_TIME_ORDER > 1 ? inner_shape[1] : 1;}
//
//	shape inner_shape;
//	shape outer_shape;
//
//	const auto InnerShape() const { return inner_shape; }
//	const auto OuterShape() const { return outer_shape; }
//
//	shape_packet<COMPILE_TIME_ORDER - 1, int*> access_packet() const {
//		return shape_packet<COMPILE_TIME_ORDER - 1, int*>(sz / inner_shape[COMPILE_TIME_ORDER - 1], inner_shape, outer_shape);
//	}
//	shape_packet<COMPILE_TIME_ORDER, int*> expression_packet() const {
//		return shape_packet<COMPILE_TIME_ORDER, int*>(sz, inner_shape, outer_shape);
//	}
//	shape_packet(int sz, int* is, int* os) : sz(sz), inner_shape(is), outer_shape(os) {}
//	shape_packet(const shape_packet<COMPILE_TIME_ORDER, int*>& p) : sz(p.sz), inner_shape(p.inner_shape), outer_shape(p.outer_shape) {}
//	shape_packet(const shape_packet<COMPILE_TIME_ORDER, int[COMPILE_TIME_ORDER]>& p) : sz(p.sz), inner_shape(&sz), outer_shape(&sz) {}
//	shape_packet(int size, const int* is, const int* os) : sz(size)  {
//		outer_shape = const_cast<int*>(os);
//		inner_shape = const_cast<int*>(is);
//	}
//
//
//	shape_packet(std::initializer_list<int> dims) {
//		const int* ary = dims.begin();
//
//		if /*constexpr*/ (COMPILE_TIME_ORDER == 0) {
//			sz = 1;
//		}
//		if /*constexpr*/ (COMPILE_TIME_ORDER == 1) {
//			inner_shape[0] = *ary;
//			sz = *inner_shape;
//		} else if /*constexpr*/  (COMPILE_TIME_ORDER  == 2) {
//			inner_shape[0] = ary[0];
//			inner_shape[1] = ary[1];
//			sz = inner_shape[0] * inner_shape[1];
//		} else {
//			sz = 1;
//			for (int i = 0; i < COMPILE_TIME_ORDER; ++i) {
//				inner_shape[i] = ary[i];
//				sz *= inner_shape[i];
//			}
//		}
//	}
//
//	~shape_packet() = default;
//};
//
//
//
//
//template<>
//struct shape_packet<0, int*> {
//public:
//	static constexpr int one = 1;
//	using shape = int*;
//	void printDimensions() const { std::cout <<"[" << 1 << "]"; std::cout << "/n"; }
//
//	int sz = 1;
//	int size() const { return 1; }
//	int rows() const { return 1;}
//	int cols() const { return 1;}
//
//	const int* InnerShape() const { return &sz; }
//	const int* OuterShape() const { return &sz; }
//
//	shape_packet() = default;
//	~shape_packet() = default;
//};
//
//}
//
//
//#endif /* SHAPE_H_ */
