#include <iostream>
template<int size>
struct begin {
	static void init_helper(double* d) {
		d[size] = size;
		std::cout << 1 + size << std::endl;

		if (size > 0)
			begin<(size > 0 ? size - 1 : 0)>::init_helper(d);
	}
};

template<int sz>
struct ary {
	double d[sz];

	void initialize() {
		begin<sz - 1>::init_helper(d);
	}

	template<int sz_end>
	void axpy(ary<sz>& ary) {
		d[sz_end] += ary.d[sz_end];

		if (sz_end > 0)
			axpy<(sz_end > 0 ? sz_end - 1 : 0)>(ary);
	}

};

void axpy(double* a, double* b, int sz) {
	for (int i = 0; i < sz; ++i) {
		a[i] += b[i];
	}
}
#include <omp.h>
#include <iostream>
#include <stdio.h>
#include "time.h"
#include "math.h"
int main() {
	std::cout << "Ss" << std::endl;

	ary<9000> a;
	a.initialize();
	ary<9000> b;
	b.initialize();

	float t;
	t = omp_get_wtime();
	printf("Calculating...(optimized)\n");
	for (int i = 0; i < 1000; ++i) {
		a.axpy<9000>(b);
	}

	printf("adding alpha, beta, alpha together (optimized code): %d\n", t);
	t = omp_get_wtime() - t;
	printf("Time: (%f seconds).\n", t, ((float) t));


	t = omp_get_wtime();
	printf("Calculating...(optimized)\n");
	for (int i = 0; i < 1000; ++i) {
		axpy(a.d, b.d, 9000);
	}

	printf("adding alpha, beta, alpha together (optimized code): %d\n", t);
	t = omp_get_wtime() - t;
	printf("Time: (%f seconds).\n", t, ((float) t));

	std::cout << "Ss" << std::endl;
}
