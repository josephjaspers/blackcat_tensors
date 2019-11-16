/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_COMMON_H_
#define BLACKCAT_COMMON_H_

#include <type_traits>

namespace BC {

#ifndef BC_DEFAULT_SYSTEM_TAG
#define BC_DEFAULT_SYSTEM_TAG host_tag
#endif

class system_tag_base {};

class host_tag : system_tag_base {
public:
	using default_floating_point_type = double;
	using default_integer_type = int;
	static const char* name() { return "BC::host_tag"; }
};

class device_tag : system_tag_base {
public:
	using default_floating_point_type = float;
	using default_integer_type = int;
	static const char* name() { return "BC::device_tag"; }
};

template<class T>
struct is_system_tag {
	static constexpr bool value = std::is_base_of<system_tag_base, T>::value;
};

using default_system_tag_t = BC_DEFAULT_SYSTEM_TAG;

}

/*
 * This file defines all global macros and compilation switches.
 */

// --------------------------------- compile options --------------------------------- //

//#define BC_NO_OPENMP                    //Disables automatic multi-threading of element-wise operations (if openmp is linked)
#define BC_CPP17                          //Enables C++17 features -- Note: constexpr if is not supported by NVCC
//#define BC_EXECUTION_POLICIES           //Enables execution policies
//#define NDEBUG 		                  //Disables runtime checks
//BC_CPP17_EXECUTION std::execution::par  //defines default execution as parallel
//BC_CPP17_EXECUTION std::execution::seq  //defines default execution as sequential
//#define BC_CPP20						  //enables C++20 features -- None: this is reserved for future, NVCC does not support cpp20 features
//#define BC_CLING_JIT 			  		  //Defines certain code based upon if we are using a cling
// --------------------------------- override macro-option s --------------------------------- //
//#define BC_INLINE_OVERRIDE <compiler_attribute>       //overloads the default inline attribute
//#define BC_SIZE_T_OVERRIDE  <integer_type>			//overloads the default size_t (default is signed int)

// ------------- define if cuda is defined ----------------- //
#ifdef __CUDACC__
#define BC_IF_CUDA(...) __VA_ARGS__
#else
#define BC_IF_CUDA(...)
#endif

// --------------------------------- inline macros -----------------------------------------//

#ifdef __CUDACC__
	#define BCHOSTDEV __host__ __device__
#else
	#define BCHOSTDEV
#endif

#ifdef BC_INLINE_OVERRIDE
	#define BCINLINE BCHOSTDEV BC_INLINE_OVERRIDE
	#define BCHOT BC_INLINE_OVERRIDE
#else
	#if defined(__GNUG__) || defined(__GNUC__) || defined(__clang__) || defined(__cling__) 
	#define BCINLINE BCHOSTDEV  inline __attribute__((always_inline)) __attribute__((hot))  //host_device inline
	#define BCHOT   		   inline __attribute__((always_inline)) __attribute__((hot))  //device-only inline

	#elif defined(_MSC_VER)
		#define BCINLINE BCHOSTDEV __forceinline
		#define BCHOT    __forceinline

	#else
		#define BCINLINE BCHOSTDEV inline
		#define BCHOT  inline
	#endif
#endif 

// --------------------------------- unique address -----------------------------------------//


#ifdef BC_CPP20
#define BC_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
#define BC_NO_UNIQUE_ADDRESS
#endif

// --------------------------------- asserts -----------------------------------------//

// Visual Studio
#ifdef _MSC_VER 
#define __PRETTY_FUNCTION__ __FUNCTION__
#endif

#include <iostream>

namespace BC {

template<class str_type>
inline void bc_assert(bool condition, str_type msg, const char* file, const char* function, int line) {
	   if (!condition) {
			std::cout << "BC_ASSERT FAILURE: " <<
		   "\nfile: " << file <<
		   "\nfunction: " << function <<
		   "\nline: " << line <<
		   "\nerror: " << msg << std::endl;

		   throw 1;
	   }
}
#define BC_ASSERT(condition, message) { BC::bc_assert(condition, message, __FILE__, __PRETTY_FUNCTION__, __LINE__); }

}

#ifdef __CUDACC__
#include <cublas.h>
namespace BC {

#define BC_CUDA_ASSERT(...) { BC_cuda_assert((__VA_ARGS__), __FILE__, __PRETTY_FUNCTION__, __LINE__); }
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


#if __has_include(<cudnn.h>)
#include <cudnn.h>
inline void BC_cuda_assert(cudnnStatus_t code, const char *file, const char* function,  int line)
{
   if (code != CUDNN_STATUS_SUCCESS)
   {
	   std::cout << "BC_CUBLAS CALL_FAILURE: " <<
	   "cudnn error: " << cudnnGetErrorString(code) <<
	   "\nfile: " << file <<
	   "\nfunction: " << function <<
	   "\tline: " << line << std::endl;
	   throw code;
   }
}
#endif
}

#endif

// --------------------------------- bc constexpr if (NVCC doesn't support cpp17) --------------------------------- //

#define BC_CONSTEXPR_IF(conditional)\
BC::traits::constexpr_if<conditional>([&]()

#define BC_CONSTEXPR_ELSE_IF(conditional) \
, BC::traits::constexpr_if<conditional>([&]()\


#define BC_CONSTEXPR_IF_END );

// --------------------------------- module body macro --------------------------------- //


#define BC_DEFAULT_MODULE_BODY(namespace_name, class_name)		\
																\
namespace BC { 													\
																\
class host_tag;													\
class device_tag;												\
																\
namespace namespace_name {									   	\
																\
	template<class SystemTag>									\
	class class_name;											\
																\
	template<class SystemTag>									\
	using implementation = class_name<SystemTag>;				\
																\
	}															\
}

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

static constexpr BC::size_t MULTITHREAD_THRESHOLD = 16384;


#ifdef __CUDACC__
	namespace {
    	static BC::size_t CUDA_BASE_THREADS = 128;
	}

    static void set_cuda_base_threads(BC::size_t nthreads) {
    	CUDA_BASE_THREADS = nthreads;
    }

    static BC::size_t get_cuda_base_threads() {
    	return CUDA_BASE_THREADS;
    }

    static  BC::size_t calculate_threads(BC::size_t sz = CUDA_BASE_THREADS) {
        return sz > CUDA_BASE_THREADS ? CUDA_BASE_THREADS : sz;
    }

    static  BC::size_t calculate_block_dim(int size) {
        return 1 + (int)(size / CUDA_BASE_THREADS);
    }
#endif


// ------------ classname ------------- //
#if defined(__GNUG__) || defined(__GNUC__)
#include <cxxabi.h>
template<class T>
inline const char* bc_get_classname_of(const T& arg) {
	int status;
	return abi::__cxa_demangle(typeid(arg).name(),0,0,&status);
}
#else
template<class T>
inline const char* bc_get_classname_of(const T& arg) {
	return typeid(arg).name();
}
#endif

//------ seperator ---------- //

inline char bc_directory_separator()
{
#if defined _WIN32 || defined __CYGWIN__
	return '\\';
#else
	return '/';
#endif
}


template<class T=char>
void print(const T& arg='\n') {
	std::cout << arg << std::endl;
}
template<class T, class... Ts>
void print(const T& arg, const Ts&... args) {
	std::cout << arg << " ";
	print(args...);
}


}


#endif /* BLACKCAT_COMMON_H_ */
