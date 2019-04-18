/*
 * BlackCat_Common.h
 *
 *  Created on: Apr 18, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_COMMON_H_
#define BLACKCAT_COMMON_H_

namespace BC {

class host_tag;
class device_tag;

}

/*
 * This file defines all global macros and compilation switches.
 */

// --------------------------------- compile options --------------------------------- //

//#define BC_NO_OPENMP                    //Disables automatic multi-threading of element-wise operations (if openmp is linked)
#define BC_CPP17                          //Enables C++17 features -- Note: constexpr if is not supported by NVCC
//#define BC_DISABLE_TEMPORARIES		  //Disables the creation of temporaries in expressions
//#define BC_EXECUTION_POLICIES           //Enables execution policies
//#define NDEBUG 		                  //Disables runtime checks
//BC_CPP17_EXECUTION std::execution::par  //defines default execution as parallel
//BC_CPP17_EXECUTION std::execution::seq  //defines default execution as sequential
//#define BC_TREE_OPTIMIZER_DEBUG	      //enables print statements for the tree evaluator. (For developers)


// --------------------------------- override macro-option s --------------------------------- //
//#define BC_INLINE_OVERRIDE <compiler_attribute>       //overloads the default inline attribute
//#define BC_SIZE_T_OVERRIDE  <integer_type>			//overloads the default size_t (default is signed int)

// --------------------------------- inline macros -----------------------------------------//

#ifdef __CUDACC__
	#define BCHOSTDEV __host__ __device__
#else
	#define BCHOSTDEV
#endif

#ifdef BC_INLINE_OVERRIDER
#define BCINLINE BCHOSTDEV  BC_INLINE_OVERRIDER
#else
#define BCINLINE BCHOSTDEV  inline __attribute__((always_inline)) __attribute__((hot))  //host_device inline
#endif

#define BCHOT   		   inline __attribute__((always_inline)) __attribute__((hot))  //device-only inline

#include <iostream>

namespace BC {

#define BC_ASSERT(condition, message) { bc_assert(condition, message, __FILE__, __PRETTY_FUNCTION__, __LINE__); }
template<class str_type>
inline void bc_assert(bool condition, str_type msg, const char* file, const char* function, int line) {
	   if (!condition) {
			std::cout << "BC_ASSERT FAILURE: " <<
		   "\nfile: " << file <<
		   "\nfunction: " << function <<
		   "\tline: " << line <<
		   "\terror: " << msg << std::endl;

		   throw 1;
	   }
}
}

#ifdef __CUDACC__
#include <cublas.h>
namespace BC {

#define BC_CUDA_ASSERT(ans) { BC_cuda_assert((ans), __FILE__, __PRETTY_FUNCTION__, __LINE__); }
inline void BC_cuda_assert(cudaError_t code, const char *file, const char* function, int line)
{
   if (code != cudaSuccess)
   {
	   std::cout << "BC_CUDA CALL_FAILURE: " <<
	   cudaGetErrorString(code) <<
	   "\nfile: " << file <<
	   "\nfunction: " << function <<
	   "\tline: " << line << std::endl;
	   throw code;
   }
}
inline void BC_cuda_assert(cublasStatus_t code, const char *file, const char* function,  int line)
{
   if (code != CUBLAS_STATUS_SUCCESS)
   {
	   std::cout << "BC_CUBLAS CALL_FAILURE: " <<
	   "cublas error: " << code <<
	   "\nfile: " << file <<
	   "\nfunction: " << function <<
	   "\tline: " << line << std::endl;
	   throw code;
   }
}
}

#endif

// --------------------------------- bc constexpr if (NVCC doesn't support cpp17) --------------------------------- //

#define BC_CONSTEXPR_IF(conditional)\
BC::meta::constexpr_if<conditional>([&]()

#define BC_CONSTEXPR_ELSE_IF(conditional) \
, BC::meta::constexpr_if<conditional>([&]()\


#define BC_CONSTEXPR_IF_END );

// --------------------------------- module body macro --------------------------------- //


#ifdef __CUDACC__	//------------------------------------------|

#define BC_DEFAULT_MODULE_BODY(namespace_name)					\
																\
namespace BC { 													\
																\
class host_tag;													\
class device_tag;												\
																\
namespace namespace_name {									   	\
																\
	template<class system_tag>									\
	using implementation =										\
			std::conditional_t<									\
				std::is_same<host_tag, system_tag>::value,		\
					Host,										\
					Device>;									\
																\
	}															\
}

#else

#define BC_DEFAULT_MODULE_BODY(namespace_name)					 \
																 \
namespace BC { 													 \
																 \
class host_tag;													 \
class device_tag;												 \
																 \
namespace namespace_name {										 \
																 \
	template<													 \
		class system_tag,										 \
		class=std::enable_if<std::is_same<system_tag, host_tag>::value> \
	>															 \
	using implementation = Host;								 \
																 \
	}															 \
}
#endif

// --------------------------------- openmp macros --------------------------------- //

#if defined(_OPENMP) && !defined(BC_NO_OPENMP)
	#define BC_OPENMP
	#define BC_omp_parallel__		_Pragma("omp parallel")
	#define BC_omp_async__(...)    	BC_omp_parallel__ {_Pragma("omp single nowait") {__VA_ARGS__ } }
	#define BC_omp_atomic__ 		_Pragma("omp atomic")
	#define BC_omp_for__ 			_Pragma("omp parallel for")
	#define BC_omp_bar__ 			_Pragma("omp barrier")
	#define __BC_CONCAT_REDUCTION_LITERAL(oper, value) omp parallel for reduction(oper:value)
	#define BC_omp_reduction__(oper, value) BC_omp_for__ reduction(oper:value)
#else
	#define BC_omp_async__(...) __VA_ARGS__
	#define BC_omp_parallel__
	#define BC_omp_atomic__
	#define BC_omp_for__
	#define BC_omp_bar__
	#define BC_omp_for_reduction__(oper, value)
#endif

// --------------------------------- constants --------------------------------- //

namespace BC {

inline void host_sync() {
#if defined(_OPENMP) && !defined(BC_NO_OPENMP) //if openmp is defined
	BC_omp_bar__
#endif
}

inline void device_sync() {
#ifdef __CUDACC__
	cudaDeviceSynchronize();
#endif
}

inline void synchronize() {
	host_sync();
	device_sync();
}

#ifndef BC_SIZE_T_OVERRIDE
//Using a signed integer is preferable
using  size_t   = int;
#else
using  size_t   = BC_SIZE_T_OVERRIDE;
#endif


#ifdef __CUDACC__
	namespace {
    	static BC::size_t CUDA_BASE_THREADS = 128;
    	static constexpr  BC::size_t MULTITHREAD_THRESHOLD = 16384;
	}
    static void set_cuda_base_threads(BC::size_t nthreads) {
    	CUDA_BASE_THREADS = nthreads;
    }

    static BC::size_t get_cuda_base_threads() {
    	return CUDA_BASE_THREADS;
    }

    static  BC::size_t blocks(int size) {
        return 1 + (int)(size / CUDA_BASE_THREADS);
    }
    static  BC::size_t threads(int sz = CUDA_BASE_THREADS) {
        return sz > CUDA_BASE_THREADS ? CUDA_BASE_THREADS : sz;
    }
#endif

}
/*
 * Include Order:
 * 	1) Dependencies of Core
 * 	2) Core
 * 	3) Things dependent upon core
 *
 */



#endif /* BLACKCAT_COMMON_H_ */
