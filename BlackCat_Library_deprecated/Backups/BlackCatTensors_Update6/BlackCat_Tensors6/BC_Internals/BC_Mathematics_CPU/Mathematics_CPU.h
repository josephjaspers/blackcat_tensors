/*
 * BC_Mathematics_CPU.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef MATHEMATICS_CPU_H_
#define MATHEMATICS_CPU_H_

#include <cmath>
#include <iostream>
#include <string>


#include "../BlackCat_Internal_GlobalUnifier.h"
namespace BC {
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
	template<typename T, typename J>
	static void fill(T& t, const J j, int sz) {
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			t[i] = j;
		}
#pragma omp barrier
	}
	template<typename T, typename J>
	static void set_heap(T *t, J *j) {
		*t = *j;
	}
	template<typename T, typename J>
	static void set_stack(T *t, J j) {
		*t = j;
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
	template<typename T, typename J>
	static void copy(T t, J j, int sz) {
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			t[i] = j[i];
		}
#pragma omp barrier
	}
	template<typename T, typename J>
	static void copy_single_thread(T t, J j, int sz) {
		for (int i = 0; i < sz; ++i) {
			t[i] = j[i];
		}
	}

	template<int a, int b>
	struct comparator {
		static constexpr int max = a > b ? a : b;
		static constexpr int min = a > b ? b : a;
	};

	template<int sz, int curr = 0>
	struct inline_copy {
		static constexpr int MAX_AUTO_INLINE = 100;

		template<class T, class U> inline //__attribute__((always_inline))
		static void copy(T t, U u) {
			t[curr] = u[curr];
			if (curr + 3 < sz) {
				t[curr + 3] = u[curr + 3];
				t[curr + 2] = u[curr + 2];
				t[curr + 1] = u[curr + 1];
			} else if (curr + 2 < sz) {
				t[curr + 2] = u[curr + 2];
				t[curr + 1] = u[curr + 1];
			} else if (curr + 1 < sz) {
				t[curr + 1] = u[curr + 1];
			}

			inline_copy<comparator<sz, MAX_AUTO_INLINE>::min, comparator<curr + 4, MAX_AUTO_INLINE>::min>::copy(t, u);
		}
	};
	template<int sz>
	struct inline_copy<sz, sz> {
		template<class T, class U>
		static void copy(T t, U u) {
			/*terminate*/
		}
	};

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

	template<int curr, int ... stack>
	struct f {
		void fill(int* ary) {
			ary[0] = curr;
			f<stack...>().fill(&ary[1]);
		}
	};
	template<int dim>
	struct f<dim> {
		void fill(int* ary) {
			ary[0] = dim;
		}
	};

private:
	template<typename T, class rank_ary>
	static void printHelper(const T ary, rank_ary ranks, int order, std::string indent, int printSpace) {
		--order;

		if (order > 1) {
			std::cout << indent << "--- --- --- --- --- " << order << " --- --- --- --- ---" << std::endl;
			auto adj_indent = indent + "    "; //add a tab to the index

			for (int i = 0; i < ranks[order]; ++i) {
				printHelper(&ary[i * ranks[order - 1]], ranks, order, adj_indent, printSpace);
			}

			auto gap = std::to_string(order);
			auto str = std::string(" ", gap.length());
			std::cout << indent << "--- --- --- --- --- " << order << " --- --- --- --- ---" << std::endl;

			//if matrix
		} else if (order == 1) {
			std::cout << indent << "--- --- --- --- --- " << order << " --- --- --- --- ---" << std::endl;

			for (int j = 0; j < ranks[order - 1]; ++j) {
				std::cout << indent + indent + "[ ";

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
	template<typename T, class rank_ary>
	static void print(const T ary, rank_ary ranks, int order, int print_length) {
		std::string indent = "";
		printHelper(ary, ranks, order, indent, print_length);

		std::cout << std::endl;
	}
	template<typename T>
	static void print(const T ary, int size, int printSpace) {
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
};

}
#endif /* MATHEMATICS_CPU_H_ */
