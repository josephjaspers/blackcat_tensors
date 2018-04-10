///*
// * BC_Tensor_Super_Shape_Inheritance.h
// *
// *  Created on: Nov 27, 2017
// *      Author: joseph
// */
//
//#ifndef BC_TENSOR_SUPER_SHAPE_INHERITANCE_H_
//#define BC_TENSOR_SUPER_SHAPE_INHERITANCE_H_
//
//template<class, class, bool>
//class Scalar;
//
//template<class, class, int...>
//class Vector;
//
//template<class, class, int, int>
//class Matrix;
//
//template<class, class, int...>
//class Tensor3;
//
//template<class, class, int...>
//class Tensor;
//
//namespace BC_Shape_Identity {
//
//	constexpr int MAX_DIMENSIONALITY = 500;
//	constexpr int MAX_DEPTH = 500;
//
//	template<int ... dims>
//	struct _shape;
//
//	namespace dimensionality_helper {
//
//		template<int curr_dim, int ... dims>
//		constexpr int rank() {
//			return 1 + rank<dims...>();
//		}
//
//		template<int curr_dim, int ... dims>
//		constexpr int valid_rank() {
//			return 1 + rank<dims...>();
//		}
//
//		template<int f, int ... remain>
//		struct first {
//			static constexpr int value = f;
//		};
//
//		template<int, class, int...>
//		struct extract_pack {};
//
////		template<int number_to_extract, int ... to, int from_first, int ... from>
////		struct extract_pack<number_to_extract, _shape<to...>, from_first, from...> {
////
////			static constexpr bool conditional = number_to_extract > 0 ? extract_pack<number_to_extract - 1, _shape<to..., from_first>, from...>::conditional : true;
////			using current_extraction = _shape<to..., from_first>;
////			using type = typename extract_pack<number_to_extract - 1, _shape<to..., from_first>>::type;
////		};
////
////		template<int ... to, int ... from>
////		struct extract_pack<0, _shape<to...>, from...> {
////
////			static constexpr bool conditional = true;
////			using current_extraction = _shape<to...>;
////			using type = current_extraction;
////		};
//
//		template<int curr_dim, int ... dims>
//		struct valid_ranks {
//			//This returns the number of ranks that do not have trailings ranks. (Ranks that end in 1s, or 0s, as they do not effect the shape)
//			static constexpr int rank = 1 + (sizeof...(dims) > 0 ? valid_ranks<curr_dim + 1, dims...>::rank : 0);
//
//			//value is true if all ranks in the set are valid
//			static constexpr bool conditional = curr_dim > 1 || (sizeof...(dims) > 0 ? valid_ranks<curr_dim + 1, dims...>::is_valid : false);
//
//			using adjusted_ranks = typename extract_pack<rank, _shape<>, dims...>::type;
//
//		};
//
//		template<class >
//		struct get_adjusted {
//
//		};
//		template<int removed, int ... to, int ... from>
//		struct get_adjusted<extract_pack<removed, _shape<to...>, from...>> {
//		};
//	}
//
//}
//
//#endif /* BC_TENSOR_SUPER_SHAPE_INHERITANCE_H_ */
