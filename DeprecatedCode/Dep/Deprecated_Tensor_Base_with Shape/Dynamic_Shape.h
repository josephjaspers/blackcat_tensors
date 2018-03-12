///*
// * Tensor_Lv3_Base.h
// *
// *  Created on: Dec 30, 2017
// *      Author: joseph
// */
//
//#ifndef TENSOR_LV3_BASE_H_
//#define TENSOR_LV3_BASE_H_
//
//#include "Tensor_Operations.h"
//#include "Tensor_Utility.h"
//namespace BC {
//
//template<
//	class scalar_type,
//
//	template<class>
//	class IDENTITY,
//	class DERIVED,
//	class MATHLIB>
//
//
//struct Dynamic_Shape;
//
//template<
//	class scalar_type,
//
//	template<class>
//	class IDENTITY,
//	class DERIVED,
//	class MATHLIB>
//
//
//struct Dynamic_Shape {
//
//private:
//
//	using primary_parent = Tensor_Math_Core<scalar_type, IDENTITY ,DERIVED, MATHLIB>;
//	using functor_type = typename primary_parent::functor_type;
//	typedef std::initializer_list<int> _shape;
//
//public:
//
//	functor_type array;
//	static constexpr int degree = IDENTITY<DERIVED>::RANK;
//	const bool rank_ownership;
//	const bool lead_ownership;
//	const bool array_ownership = rank_ownership && lead_ownership;
//
//	int sz = 0;
//	int* ranks;
//	int* ld;
//
//
//
//public:
//	int size()  const { return sz; }
//	int order() const { return degree; }
//
//	const auto& getShape() const { return ranks; }
//	const auto& getLD()    const { return ld;	 }
//	auto& getShape() { return ranks; }
//	auto& getLD()    { return ld; }
//
//	void init() {
//		MATHLIB::initialize(ranks, degree);
//		MATHLIB::initialize(ld, degree);
//	}
//
//
//		//Constructor for non-subTensor (default ld)
//		Dynamic_Shape(std::initializer_list<int> set) : rank_ownership(true), lead_ownership(true) {
//			init();
//			ranks[0] = set.begin()[0];
//			ld[0] = set.begin()[0];
//			sz = set.begin()[0];
//			for (int i = 1; i < set.size(); ++i) {
//				ranks[i] = set.begin()[i];
//				ld[i] = set.begin()[i] * ld[i - 1];
//				sz *= ranks[i];
//			}
//			MATHLIB::initialize(array, sz);
//		}
//		//Constructor for sub-tensor (not an index tensor)
//		template<class... params>
//		Dynamic_Shape(std::initializer_list<int> set, const int* lead_dimensions, const params&... p)
//		: lead_ownership(false), rank_ownership(true), array(p...) {
//
//			ld = lead_dimensions;
//			MATHLIB::initialize(ranks, set.size());
//			ranks[0] = set.begin()[0];
//			for (int i = 1; i < set.size(); ++i) {
//				ranks[i] = set.begin()[i];
//				sz *= ranks[i];
//			}
//			MATHLIB::initialize(array, sz);
//		}
//
//		//Constructor for index tensor OR expression tensor
//		template<class... params>
//		Dynamic_Shape(int* dims, int* leads, const params&... p)
//		: rank_ownership(false), lead_ownership(false), array(p...) {
//
//			sz = 1;
//			for (int i = 0; i < degree; ++i)
//				sz *= dims[i];
//
//			ranks = dims;
//			ld = leads;
//		}
//};
//
//}
//
//
//#endif /* TENSOR_LV3_BASE_H_ */
