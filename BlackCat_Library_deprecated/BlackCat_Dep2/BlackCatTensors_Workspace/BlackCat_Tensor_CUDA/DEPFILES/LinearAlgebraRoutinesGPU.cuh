///*
// * LinearAlgebraRoutinesGPU.cuh
// *
// *  Created on: Sep 13, 2017
// *      Author: joseph
// */
//
//#ifndef LINEARALGEBRAROUTINESGPU_CUH_
//#define LINEARALGEBRAROUTINESGPU_CUH_
//
//// includes CUDA Runtime
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include "device_launch_parameters.h"
//#include <iostream>
//
//
//template<typename number_type>
//class Tensor_OperationsGPU{
//public:
//	//General
//    //Memory Management [written for preparation for writing with CUDA]]
//    static void initialize	(number_type*& d, unsigned sz) {
//    	cudaMalloc((void **)d, sizeof(number_type) * sz);
//    }
//    static void destruction	(number_type* d) {
//    	cudaFree(d);
//    }
//
//    static void correlation(number_type* s, const number_type* filter, const number_type* signal, unsigned sz);
//
//    //not kernal
//    static void correlation(number_type* s, unsigned order, const unsigned* ranks,const number_type* filter, const unsigned* f_ld,
//       																		      const number_type* signal, const unsigned* s_ld);
//       //dimensional --movement correlation
//       static void cross_correlation(number_type* s, unsigned cor_mv, unsigned order, const  unsigned* store_ld, const number_type* filter,const  unsigned * f_ld,const  unsigned* f_ranks,
//       																										  const number_type* signal, const unsigned * s_ld, const unsigned* s_ranks);
//
//
//       static void print		(const number_type* ary, const unsigned* dims, const unsigned* lead_dims, unsigned index) {
//
//    	   unsigned sz = 1;
//    	   for (unsigned i = 0; i < index; ++i) {
//    		   sz *= dims[i];
//    	   }
//    	   number_type* host_data = new number_type[sz];
//    	   cudaMemcpy(&ary,&host_data,sizeof(number_type) * sz,cudaMemcpyHostToDevice);
//
//        	if (index < 3) {
//        		for (unsigned r = 0; r < dims[0]; ++r) {
//
//        			if (r != 0)
//        			std::cout << std::endl;
//
//        			for (unsigned c = 0; c< dims[1]; ++c) {
//        				auto str =std::to_string(ary[r + c * lead_dims[index - 1]]);
//        				str = str.substr(0, str.length() < 5 ? str.length() : 5);
//        				std::cout << str << " ";
//        			}
//        		}
//        		std::cout << "]" << std::endl << std::endl;
//
//        	} else {
//        		std::cout << "[";
//        		for (unsigned i = 0; i < dims[index - 1]; ++i) {
//        			print(&ary[i * lead_dims[index - 1]], dims, lead_dims, index - 1);
//        		}
//        	}
//        	delete[] host_data;
//        }
//       //Pointwise ---- THESE ARE THE KERNALS
//
//       //Indexing
//       static void max_val		(const number_type* m, number_type* max_val, unsigned sz);
//       static void min_val		(const number_type* m, number_type* min_val, unsigned sz);
//       static void max_index	(const number_type * m, number_type * max_val, unsigned * index, unsigned sz);
//       static void min_index	(const number_type * m, number_type * min_val, unsigned * index, unsigned sz);
//
//       static void dot		(number_type* store, unsigned store_LD, const number_type* m1, unsigned m1_row, unsigned m1_col, unsigned m1_inc,
//       															const number_type* m2, unsigned m2_row, unsigned m2_col, unsigned m2_inc);
////kernal ended ------------------------------------------------------------------------
//       //Pointwise increments ------------- COL Major --- for degree > 2 Ten
//       static void copy     (number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1, const unsigned* m1_LD);
//       static void fill     (number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, number_type m1);
//       static void transpose(number_type* s, unsigned s_ld, const number_type* m, unsigned r, unsigned c, unsigned m_ld);
//
//       static void power	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD);
//       static void multiply(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD);
//       static void divide	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD);
//       static void add		(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD);
//       static void subtract(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type* m2,  const unsigned* m2_LD);
//       //By Scalar
//       static void power	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal);
//       static void multiply(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal);
//       static void divide	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal);
//       static void add		(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal);
//       static void subtract(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,  const unsigned* m1_LD, const number_type scal);
//};
////#endif /* LINEARALGEBRAROUTINESGPU_CUH_ */
