/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   LinearAlgebraRoutines.h
 * Author: joseph
 *
 * Created on July 23, 2017, 9:11 PM
 */

#ifndef LINEARALGEBRAROUTINES_H
#define LINEARALGEBRAROUTINES_H

#include <iostream>
#include <cmath>
#include <string>

#include <math.h>
#include "cblas.h"

class CPU {
public:
	/*
	 * T -- An object (Generally an array) with a functional [] operator
	 * J -- Either a value on the stack to be assigned or another array similar to above.
	 *
	 * J -- may also be a single value passed by pointer, certains methods are overloaded to handle these instance
	 *
	 */

	//destructor - no destructor are for controlling destruction of the pointer
	template<typename T>
	static void initialize(T*& t, int sz) {
		t = new T[sz];
	}
	template<typename T>
	static void unified_initialize(T*& t, int sz) {
		t = new T[sz];
	}
	template<typename T>
	static void destroy(T* t) {
		delete[] t;
	}

	template<class T, class U>
	static void set(T* scalar, U value) {
		*scalar = value;
	}
	template<class T, class U>
	static void set(T* scalar, const U* value) {
		*scalar = *value;
	}

	template<typename T, typename J>
	static void fill(T& t, const J j, int sz) {
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			t[i] = j;
		}
#pragma omp barrier
	}
	template<typename T, typename J>
	static void fill(T& t, const J* j, int sz) {
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			t[i] = *j;
		}
#pragma omp barrier

	}
	template<typename T>
	static void zero(T& t, int sz) {
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			t[i] = 0;
		}
#pragma omp barrier
	}

	template<int sz, class T, class J>
	struct optimized {
		inline __attribute__((always_inline)) static void copy(T t, const J j) {
			t[sz] = j[sz];
			t[sz - 1] = j[sz - 1];
			optimized<sz - 2, T, J>::copy(t, j);
//			t[sz - 2] = j[sz - 2];
//			t[sz - 3] = j[sz - 3];
//			t[sz - 4] = j[sz - 4];
//			t[sz - 5] = j[sz - 5];
//			t[sz - 6] = j[sz - 6];
//			t[sz - 7] = j[sz - 7];
//
//			optimized<sz - 8, T, J>::copy(t, j);
		}
	};

	template<class T, class J>
	struct optimized<0, T, J> {
		inline __attribute__((always_inline)) static void copy(T t, const J j) {
			t[0] = j[0];
		}
	};

	template<int sz, class T, class J>
	struct optimized2 {

		inline __attribute__((always_inline)) static void copy(T t, const J j, int i) {
			optimized2<sz / 4, T, J>::copy(t, j, i);
			optimized2<sz / 4, T, J>::copy(t, j, i + sz / 4);
			optimized2<sz / 4, T, J>::copy(t, j, i + (sz * 2 / 4));
			optimized2<sz / 4, T, J>::copy(t, j, i +  (sz * 3 / 4));

		}
	};
	template<class T, class J>
	struct optimized2<2, T, J> {

		inline __attribute__((always_inline)) static void copy(T t, const J j, int i) {
			t[i] = j[i];
			t[i + 1] = j[i + 1];
		}
	};

	template<class T, class J>
	struct optimized2<8, T, J> {

		inline __attribute__((always_inline)) static void copy(T t, const J j, int i) {
			t[i] = j[i];
			t[i + 1] = j[i + 1];
			t[i + 2] = j[i + 2];
			t[i + 3] = j[i + 3];
			t[i + 4] = j[i + 4];
			t[i + 5] = j[i + 5];
			t[i + 6] = j[i + 6];
			t[i + 7] = j[i + 7];
		}
	};

	template<class T, class J>
	struct optimized2<4, T, J> {

		inline __attribute__((always_inline)) static void copy(T t, const J j, int i) {
			t[i] = j[i];
			t[i + 1] = j[i + 1];
			t[i + 2] = j[i + 2];
			t[i + 3] = j[i + 3];

		}
	};

	template<typename T, typename J>
	static void opt_copy(T t, const J j, int sz) {
		const int eval_distance = 512;
#pragma omp parallel for
		for (int i = 0; i < sz - eval_distance; i += eval_distance) {
			optimized2<eval_distance, T, J>::copy(t, j, 0);
		}
		for (int i = sz - eval_distance; i < sz; ++i) {
			t[i] = j[i];
		}
#pragma omp barrier
	}

	template<typename T, typename J>
	static void copy(T t, const J j, int sz) {
		opt_copy(t, j, sz);
//#pragma omp parallel for
//		for (int i = 0; i < sz; ++i) {
//			t[i] = j[i];
//		}
//#pragma omp barrier
	}

	template<typename T, typename J>
	static void copy_stack(T& t, J j, int sz) {
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			t[i] = j[i];
		}
#pragma omp barrier
	}
	template<typename T, typename J>
	static void evaluate(T& t, const J& j, int sz) {
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			t[i] = j[i];
		}
#pragma omp barrier
	}
	template<typename T, typename J>
	static void randomize(T t, J lower_bound, J upper_bound, int sz) {
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			t[i] = ((double) (rand() / ((double) RAND_MAX + 1)) * (upper_bound - lower_bound)) + lower_bound;

		}
#pragma omp barrier
	}
	template<typename T, typename J>
	static void sum(T* sum, J ary, int sz) {
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			*sum += ary[i];
		}
#pragma omp barrier
	}

	template<typename T>
	static void transpose(T * s, unsigned s_ld, const T * m, unsigned rows, unsigned cols, unsigned m_ld) {

		for (unsigned r = 0; r < rows; ++r) {
			for (unsigned c = 0; c < cols; ++c) {
				s[r * s_ld + c] = m[c * m_ld + r];
			}
		}
	}
	template<class T, class U>
	static void transpose(T * s, const U * m, unsigned new_rows, unsigned new_cols) {

		for (unsigned r = 0; r < new_cols; ++r) {
			for (unsigned c = 0; c < new_rows; ++c) {
				s[c * new_cols + r] = m[c + r * new_rows];
			}
		}
	}

private:
	template<typename T>
	static void printHelper(const T ary, const int* ranks, int order, std::string indent, int printSpace) {
		--order;

		if (order > 1) {
			std::cout << indent << "--- --- --- --- --- (" << order + 1 << ") --- --- --- --- ---" << std::endl;
			indent += "    "; //add a tab to the index

			for (int i = 0; i < ranks[order]; ++i) {
				printHelper(&ary[i * ranks[order - 1]], ranks, order, indent, printSpace);
			}

			auto gap = std::to_string(order);
			auto str = std::string(" ", gap.length());
			std::cout << indent << "--- --- --- --- --- " << str << " --- --- --- --- ---" << std::endl;

		} else if (order == 1) {
			std::cout << indent << "--- --- --- --- --- (" << order + 1 << ") --- --- --- --- ---" << std::endl;

			for (int j = 0; j < ranks[order - 1]; ++j) {
				std::cout << "[ ";

				for (int i = 0; i < ranks[order]; ++i) {
					//convert to string --- seems to not be working with long/ints?
					auto str = std::to_string(ary[i * ranks[order - 1] + j]);

					//if the string is longer than the printspace truncate it
					str = str.substr(0, str.length() < printSpace ? str.length() : printSpace);

					//if its less we add blanks (so all numbers are uniform in size)
					if (str.length() < printSpace)
						str += std::string(" ", printSpace - str.length());

					//print the string
					std::cout << str;

					//add some formatting fluff
					if (i < ranks[order] - 1)
						std::cout << " | ";

				}
				std::cout << " ]";
				std::cout << std::endl;
			}
		} else {
			std::cout << "[ ";
			for (int i = 0; i < ranks[order]; ++i) {
				//convert to string --- seems to not be working with long/ints?
				auto str = std::to_string(ary[i]);

				//if the string is longer than the printspace truncate it
				str = str.substr(0, str.length() < printSpace ? str.length() : printSpace);

				//if its less we add blanks (so all numbers are uniform in size)
				if (str.length() < printSpace)
					str += std::string(" ", printSpace - str.length());

				//print the string
				std::cout << str;

				//add some formatting fluff
				if (i < ranks[order] - 1)
					std::cout << " | ";
			}
			std::cout << " ]";
		}
	}
public:
	template<typename T>
	static void print(const T ary, const int* ranks, int order, int print_length) {
		std::string indent = "";
		printHelper(ary, ranks, order, indent, print_length);

		std::cout << std::endl;
	}

	template<typename T>
	static void print(const T ary, int size, unsigned printSpace) {
		std::cout << "[ ";
		for (int i = 0; i < size; ++i) {
			auto str = std::to_string(ary[i]);
			str = str.substr(0, str.length() < printSpace ? str.length() : printSpace);
			std::cout << str;

			if (i < size - 1) {
				std::cout << " | ";
			}

		}
		std::cout << " ]" << std::endl;

	}
	template<typename T>
	static void print(T ary, int size) {
		print(ary, size, 5);
	}

	template<typename T>
	static void sigmoid(T ary, int sz) {
		for (int i = 0; i < sz; ++i) {
			ary[i] = 1 / (1 + std::pow(2.71828, -ary[i]));
		}
	}
	/*
	 * a = M x N
	 * b = N x K
	 * c = M x K
	 */

	template<class A, class B, class C>
	static void matmul(C c, A a, int m, int n, B b, int k) {
#pragma omp parallel for
		for (int z = 0; z < k; ++z) {
			for (int x = 0; x < n; ++x) {
				for (int y = 0; y < m; ++y) {
					c[z * m + y] += a[x * m + y] * b[z * n + x];

				}
			}
		}
#pragma omp barrier
	}

	template<class T>
	static void matmulBLAS(T* c, const T* a, int m, int n, const T* b, int k) {

#pragma omp parallel for
		for (int z = 0; z < k; ++z) {
			for (int x = 0; x < n; ++x) {
				for (int y = 0; y < m; ++y) {
					c[z * m + y] += a[x * m + y] * b[z * n + x];

				}
			}
		}
#pragma omp barrier
	}
	static void matmulBLAS(double* c, const double* a, int m, int n, const double* b, int k);
	static void matmulBLAS(float* c, const float* a, int m, int n, const float* b, int k);
	static void matmulBLAS(double* c, const double* a, bool a_transposed, int m, int n, const double* b, bool b_transposed, int k);
	static void matmulBLAS(float* c, const float* a, bool a_transposed, int m, int n, const float* b, bool b_transposed, int k);

};

#include "cblas.h"

void CPU::matmulBLAS(double* c, const double* a, int m, int n, const double* b, int k) {

	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, k, n, 1, a, m, b, n, 1, c, m);
}
void CPU::matmulBLAS(float* c, const float* a, int m, int n, const float* b, int k) {

	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, k, n, 1, a, m, b, n, 1, c, m);
}

auto trans(bool b) {
	return b ? CblasTrans : CblasNoTrans;
}

void CPU::matmulBLAS(double* c, const double* a, bool a_t, int m, int n, const double* b, bool b_t, int k) {

	cblas_dgemm(CblasColMajor, trans(a_t), trans(b_t), m, k, n, 1, a, m, b, n, 1, c, m);
}
void CPU::matmulBLAS(float* c, const float* a,  bool a_t, int m, int n, const float* b, bool b_t, int k) {

	cblas_sgemm(CblasColMajor, trans(a_t), trans(b_t), m, k, n, 1, a, m, b, n, 1, c, m);
}

#endif /* LINEARALGEBRAROUTINES_H */
