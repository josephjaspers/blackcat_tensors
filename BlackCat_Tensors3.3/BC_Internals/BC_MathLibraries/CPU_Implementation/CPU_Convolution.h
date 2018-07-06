/*
 * CPU_Convolution.h
 *
 *  Created on: Jul 5, 2018
 *      Author: joseph
 */

#ifndef CPU_CONVOLUTION_H_
#define CPU_CONVOLUTION_H_

namespace BC {

struct CPU;
namespace internal {
template<int dimension, class scalar_t,class allocator> class Array;
}

template<int dimension, class scalar_t>
using CPU_Array = BC::internal::Array<dimension, scalar_t, CPU>;


template<class core_lib>
struct CPU_Convolution {

	//2 dimensional image2col accepts a matrix (in) as the image and a 4d matrix out reshaped for the output matrix(rows, cols, row_positions, col_positions)
	template<class img_out, class img_in>
	static std::enable_if_t<img_in::DIMS() == 2> img2col(img_out out, img_in in) {
		static_assert(img_out::DIMS() == 4 && img_in::DIMS() == 2, "img2col 2d requires a 4d and 2d tensors");
		//number of column positions
		for (int cp = 0; cp < out.dimension(3); ++cp) {

			//number of row positions
			for (int rp = 0; rp < out.dimension(2); ++rp) {

				//number of kernel cols
				for (int c = 0; c < out.dimension(1); ++c) {

					//number of kernel rows
					for (int r = 0; r < out.dimension(0); ++r) {
						out(r, c, rp, cp) = in(r + cp, c + rp);
					}
				}

			}
		}
	}
	template<class img_out, class img_in>
	static std::enable_if_t<img_in::DIMS() == 3> img2col(img_out out, img_in in) {
		static_assert(img_out::DIMS() == 5 && img_in::DIMS() == 3, "img2col 2d requires a 4d and 2d tensors");
		//number of depth positions
		for (int dp = 0; dp < out.dimension(3); ++dp) {
			//number of columns positions
			for (int cp = 0; cp < out.dimension(2); ++cp) {
				//number of row positions
				for (int rp = 0; rp < out.dimension(1); ++rp) {
					for (int d = 0; d < out.dimension(0); ++d)
						for (int c = 0; c < out.dimension(0); ++c)
							for (int r = 0; r < out.dimension(0); ++r)
								out(r,c,d,rp,cp,dp) = in(r + cp, c + rp, d + dp);
				}

			}
		}
	}

	template<class scalar_t>
	static void conv2d(CPU_Array<2, scalar_t> out, CPU_Array<2, scalar_t> in, CPU_Array<2, scalar_t> filter) {
		int rpos = in.rows() - filter.rows() + 1;
		int cpos = in.cols() - filter.cols() + 1;
		int gemv_cols = rpos * cpos;

		CPU_Array<4, scalar_t> toeplitz(BC::Shape<4>(filter.rows(), filter.cols(), rpos, cpos));
		img2col(toeplitz, in);

		static constexpr scalar_t alpha_mod = 1;
		static constexpr scalar_t beta_mod = 0;


		core_lib::gemv(true, filter.size(), gemv_cols, &alpha_mod, toeplitz, in.ld1(), filter, 1, &beta_mod, out, 1);
	}

//	template<class scalar_t>
//	static void conv2d(CPU_Array<3, scalar_t> out, CPU_Array<3, scalar_t> in, CPU_Array<3, scalar_t> filter) {
//		int rpos = in.rows() - filter.rows() + 1;
//		int cpos = in.cols() - filter.cols() + 1;
//		int dpos = in.dimension(2) - filter.dimension(2) + 1;
//		int gemv_cols = rpos * cpos * dpos;
//
//		CPU_Array<5, scalar_t> toeplitz(BC::Shape<5>(filter.rows(), filter.cols(), filter.dimension(2), rpos, cpos, dpos));
//		img2col(toeplitz, in);
//
//		static constexpr scalar_t alpha_mod = 1;
//		static constexpr scalar_t beta_mod = 0;
//
//
//		core_lib::gemv(true, filter.size(), gemv_cols, &alpha_mod, toeplitz, in.ld1(), filter, 1, &beta_mod, out, 1);
//	}

};

}






#endif /* CPU_CONVOLUTION_H_ */
