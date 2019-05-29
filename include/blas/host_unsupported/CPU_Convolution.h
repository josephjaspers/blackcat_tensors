///*
// * CPU_Convolution.h
// *
// *  Created on: Jul 5, 2018
// *      Author: joseph
// */
//
//#ifndef CPU_CONVOLUTION_H_
//#define CPU_CONVOLUTION_H_
//
//namespace BC {
//struct CPU;
//namespace exprs     {
//template<int dimension, class value_type,class allocator> class Array;
//}
//
//template<int dimension, class value_type>
//using CPU_Array = BC::exprs::Array<dimension, value_type, CPU>;
//
//
//template<class core_lib>
//struct CPU_Convolution {
//
//    //2d tensor for 2d conv
//    template<class img_out, class img_in>
//    static std::enable_if_t<img_in::tensor_dimension == 2> img2col(img_out out, img_in in, BC::size_t  padding = 0,  BC::size_t  stride = 1) {
//        static_assert(img_out::tensor_dimension == 4 && img_in::tensor_dimension == 2, "img2col 2d requires a 4d and 2d tensors");
//
//        if (padding == 0)
//            for (int cp = 0; cp < out.dimension(3); ++cp)         //number of column positions
//                for (int rp = 0; rp < out.dimension(2); ++rp)         //number of column positions
//                    for (int c = 0; c < out.dimension(1); ++c)                 //number  of kernel cols
//                        for (int r = 0; r < out.dimension(0); ++r)                     //number of kernel rows
//                            out(r, c, rp, cp) = in(r + rp * stride, c + cp * stride);
//        else
//            for (int cp = 0; cp < out.dimension(3); ++cp)         //number of column positions
//                for (int rp = 0; rp < out.dimension(2); ++rp)         //number of column positions
//                    for (int c = 0; c < out.dimension(1); ++c)                 //number of kernel cols
//                        for (int r = 0; r < out.dimension(0); ++r)     {                //number of kernel rows
//
//                            BC::size_t  r_index = r + rp * stride - padding;
//                            BC::size_t  c_index = c + cp * stride - padding;
//
//                            if ((r_index >= 0) && (c_index >= 0) && (r_index < in.rows()) && (c_index < in.cols()))
//                                out(r, c, rp, cp) = in(r_index, c_index);
//                            else
//                                out(r, c, rp, cp) = 0;
//                        }
//    }
//
//
//    //3d tensor for 2d conv
//    template<class img_out, class img_in>
//    static std::enable_if_t<img_in::tensor_dimension == 3> img2col(img_out out, img_in in, BC::size_t  padding = 0,  BC::size_t  stride = 1) {
//        static_assert(img_out::tensor_dimension == 5 && img_in::tensor_dimension == 2, "img2col 2d requires a 4d and 2d tensors");
//
//        if (padding == 0)
//            for (int cp = 0; cp < out.dimension(4); ++cp)         //number of column positions
//                for (int rp = 0; rp < out.dimension(3); ++rp)         //number of column positions
//                    for (int d = 0; d < out.dimension(2); ++d)
//                        for (int c = 0; c < out.dimension(1); ++c)                 //number  of kernel cols
//                            for (int r = 0; r < out.dimension(0); ++r)                     //number of kernel rows
//                                out(r, c, d, rp, cp) = in(r + rp * stride, c + cp * stride, d);
//        else
//            for (int cp = 0; cp < out.dimension(4); ++cp)         //number of column positions
//                for (int rp = 0; rp < out.dimension(3); ++rp)         //number of column positions
//                    for (int d = 0; d < out.dimension(2); ++d)
//                        for (int c = 0; c < out.dimension(1); ++c)                 //number of kernel cols
//                            for (int r = 0; r < out.dimension(0); ++r)     {                //number of kernel rows
//
//                                BC::size_t  r_index = r + rp * stride - padding;
//                                BC::size_t  c_index = c + cp * stride - padding;
//
//                                if ((r_index >= 0) && (c_index >= 0) && (r_index < in.rows()) && (c_index < in.cols()))
//                                    out(r, c, d, rp, cp) = in(r_index, c_index, d);
//                                else
//                                    out(r, c, d, rp, cp) = 0;
//                        }
//    }
//
//
//
//    template<class value_type>
//    static void conv2(CPU_Array<2, value_type> out, CPU_Array<2, value_type> in, CPU_Array<2, value_type> filter, BC::size_t  padding = 0, BC::size_t  stride = 1) {
//        BC::size_t  rpos = (in.rows() - filter.rows()) / stride + 1 + padding * 2;
//        BC::size_t  cpos = (in.cols() - filter.cols()) / stride + 1 + padding * 2;
//        BC::size_t  k_length = rpos * cpos;
//
//        //assert_valid_dims
//        if (rpos != out.rows()) {
//            std::cout << " output matrix row_length mismatch: param" << out.rows() << " != " << rpos << std::endl;
//            throw std::invalid_argument("incorrect parameter");
//        }
//        if (cpos != out.cols()) {
//            std::cout << " output matrix col_length mismatch: param" << out.cols() << " != " << cpos << std::endl;
//            throw std::invalid_argument("incorrect parameter");
//        }
//        if (rpos * cpos != out.size()) {
//            std::cout << " output matrix size mismatch: param" << out.size() << " != " << rpos * cpos << std::endl;
//            throw std::invalid_argument("incorrect parameter");
//        }
//
//        CPU_Array<4, value_type> toeplitz(BC::Shape<4>(filter.rows(), filter.cols(), rpos, cpos));
//        img2col(toeplitz, in, padding, stride);
//
//        static constexpr value_type alpha_mod = 1;
//        static constexpr value_type beta_mod = 0;
//
//        core_lib::gemv(true, filter.size(), k_length, &alpha_mod, toeplitz, in.leading_dimension(1), filter, 1, &beta_mod, out, 1);
//    }
//
//
//    template<class value_type>
//        static void conv2(CPU_Array<2, value_type> out, CPU_Array<3, value_type> in, CPU_Array<3, value_type> filter, BC::size_t  padding = 0, BC::size_t  stride = 1) {
//            BC::size_t  rpos = (in.rows() - filter.rows()) / stride + 1 + padding * 2;
//            BC::size_t  cpos = (in.cols() - filter.cols()) / stride + 1 + padding * 2;
//            BC::size_t  k_length = rpos * cpos * filter.dimension(2);
//
//            //assert_valid_dims
//            if (rpos != out.rows()) {
//                std::cout << " output matrix row_length mismatch: param" << out.rows() << " != " << rpos << std::endl;
//                throw std::invalid_argument("incorrect parameter");
//            }
//            if (cpos != out.cols()) {
//                std::cout << " output matrix col_length mismatch: param" << out.cols() << " != " << cpos << std::endl;
//                throw std::invalid_argument("incorrect parameter");
//            }
//            if (k_length != out.size()) {
//                std::cout << " output matrix size mismatch: param" << out.size() << " != " << k_length << std::endl;
//                throw std::invalid_argument("incorrect parameter");
//            }
//
//            CPU_Array<5, value_type> toeplitz(BC::Shape<4>(filter.rows(), filter.cols(), filter.dimension(2), rpos, cpos));
//            img2col(toeplitz, in, padding, stride);
//
//            static constexpr value_type alpha_mod = 1;
//            static constexpr value_type beta_mod = 0;
//
//            core_lib::gemv(true, filter.size(), k_length, &alpha_mod, toeplitz, in.leading_dimension(1), filter, 1, &beta_mod, out, 1);
//        }
//
//
//
//
//};
//
//}
//
//
//
//
//
//
//#endif /* CPU_CONVOLUTION_H_ */
