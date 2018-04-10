#include "BlackCat_Tensors.h"
#include <iostream>
#include <omp.h>
#include "math.h"
#include "time.h"
void add(double* a, double* b, double* c, int sz) {
#pragma omp parallel for

	for (int i = 0; i < sz; ++i) {
		a[i] = b[i] + c[i];
	}

#pragma omp barrier
}
void add(double* a, double* b, double* c, double* d, int sz) {
#pragma omp parallel for

	for (int i = 0; i < sz; ++i) {
		a[i] = b[i] + c[i] + d[i];
	}
#pragma omp barrier
}

namespace operations {

	struct add {
		template<class l, class r>
		auto calc(l left, r right) {
			return left + right;
		}
	};
	struct sub {
		template<class l, class r>
		auto calc(l left, r right) {
			return left - right;
		}
	};
	struct mul {
		template<class l, class r>
		auto calc(l left, r right) {
			return left * right;
		}
	};

}
//ORDER 1
#include <iostream>
template<class lv, class rv>
struct expression {

	lv left;
	rv right;
	expression(lv l, rv r) :
			left(l), right(r) {
	}
	__attribute__((always_inline)) auto operator [](int index) {
		return (left[index] + right[index]);
	}


	template<typename T>
	void eval(T eval_to, int sz) {
#pragma omp parallel for
		for (int i = 0; i < sz; ++i) {
			eval_to[i] = left[i] + right[i];
		}
#pragma omp barrier
	}
};



template<class T, int sz>
struct ary {
	T* array;

	ary<T, sz>(T* a) {
		this->array = a;
	}

	ary<T, sz>() {
		array = new T[sz];
	}

	expression<T*, T*> operator +(const ary<T, sz>& other_ary) {
		return expression<T*, T*>(array, other_ary.array);
	}

	ary<T, sz>& operator =(expression<T*, T*> exp) {

		for (int i = 0; i < sz; ++i) {
			array[i] = exp.left[i] + exp.right[i];
		}
		return *this;
	}

	~ary() {
		delete[] array;
	}
};

int SpeedTests() {

//	omp_set_num_threads(6);

	Matrix<double> alpha(1000, 1000);
	Matrix<double> beta(1000, 1000);
	Matrix<double> gamma(1000, 1000);

	alpha.randomize(-100, 100);
	beta.randomize(-100, 100);

	float t;
	t = omp_get_wtime();
	printf("Calculating...(optimized)\n");
	for (int i = 0; i < 1000; ++i) {

		gamma = alpha + beta + beta +beta;
	}

	printf("adding alpha, beta, alpha together (optimized code): %d\n", t);
	t = omp_get_wtime() - t;
	printf("Time: (%f seconds).\n", t, ((float) t));

//
	double* c = gamma.accessor().getData();
	double* a = alpha.accessor().getData();
	double* b = beta.accessor().getData();

	int sizer = gamma.size();
	t = omp_get_wtime();

	std::cout << std::endl << std::endl;
	printf("adding alpha, beta, alpha, than alpha again (\"generic dumb version\") \n");

	for (int i = 0; i < 1000; ++i) {
		add(c, a, b, sizer);
	}
	t = omp_get_wtime() - t;
	printf("Time: (%f seconds).\n", t, ((float) t));

	std::cout << "success " << std::endl;

	std::cout << std::endl << std::endl;
	printf("Best possible version (hardcoded)  \n");
	t = omp_get_wtime();

	for (int i = 0; i < 1000; ++i) {
		add(c, a, b, sizer);
	}
	t = omp_get_wtime() - t;
	printf("Time: (%f seconds).\n", t, ((float) t));

	ary<double, 1000000> eval(c);
	ary<double, 1000000> lef(a);
	ary<double, 1000000> rig(b);

	std::cout << std::endl << std::endl;
	printf("alt (hardcoded)  \n");
	t = omp_get_wtime();

	for (int i = 0; i < 1000; ++i) {
		eval = lef + rig;
	}
	t = omp_get_wtime() - t;
	printf("Time: (%f seconds).\n", t, ((float) t));

	std::cout << std::endl << std::endl;
	std::cout << "success " << std::endl;
}

int main() {

	SpeedTests();
}
