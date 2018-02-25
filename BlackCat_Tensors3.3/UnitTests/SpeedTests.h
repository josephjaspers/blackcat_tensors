#include <iostream>
#include <omp.h>
#include "math.h"
#include "time.h"
#include <omp.h>
#include "../BlackCat_Tensors.h"
#include "BlackCat_TensorFunctions.cu"

using BC::Vector;
using BC::Matrix;

using namespace BC::NN_Functions;


struct x2 {

	template<class l>
	auto operator ()(l left) const { return std::pow(left, 2); }
};

template<int SIZE, int repetitions>
int speedTestDelayedEval() {
	const int reps = repetitions;

	using vec = Vector<float>;

	vec a(SIZE);
	vec b(SIZE);
	vec c(SIZE);
	vec d(SIZE);
	vec e(SIZE);

	a.randomize(0, 100);
	b.randomize(0, 100);
	c.randomize(0, 100);
	d.randomize(0, 100);
	e.randomize(0, 100);

	float t;

	t = omp_get_wtime();

	for (int i = 0; i < reps; ++i) {
		a = ((e == b ** c) + d);
	}
	t = omp_get_wtime() - t;

	std::cout <<"Lazy = " << SIZE << "    " << t << std::endl;

	t = omp_get_wtime();

	for (int i = 0; i < reps; ++i) {
		a = ((e = b ** c) + d);
	}

	t = omp_get_wtime() - t;


	std::cout <<"Inst = " << SIZE << "    " << t  << "\n" << std::endl;

	return 0;
}


template<int SIZE, int repetitions>
int speedTestCombine() {
	const int reps = repetitions;

	using vec = Vector<float>;

	vec x(SIZE);
	vec y(SIZE);
	vec z(SIZE);


	vec a(SIZE);
	vec b(SIZE);
	vec c(SIZE);
	vec d(SIZE);
	vec e(SIZE);
	vec f(SIZE);
	vec g(SIZE);
	vec h(SIZE);
	vec i(SIZE);

	a.randomize(0, 100);
	b.randomize(0, 100);
	c.randomize(0, 100);
	d.randomize(0, 100);
	e.randomize(0, 100);
	f.randomize(0, 100);

	g.randomize(0, 100);
	h.randomize(0, 100);
	i.randomize(0, 100);

	float t;

	t = omp_get_wtime();

	for (int iter = 0; iter < reps; ++iter) {
		x = a + b;
		y = e;
	}

	t = omp_get_wtime() - t;


	std::cout <<"sz = " << SIZE << t << std::endl;

	return 0;
}


template<int SIZE, int repetitions>
int speedTestsScalarIteration() {
	const int reps = repetitions;

	using vec = Vector<float>;
	using mat = Matrix<float>;

	vec b(SIZE);
	mat a(1,SIZE);

	a.randomize(0, 100);
	b.randomize(0, 100);

	float t;

	t = omp_get_wtime();

	for (int iter = 0; iter < reps; ++iter) {
		a[iter];
	}

	t = omp_get_wtime() - t;

	float t2;

	t2 = omp_get_wtime();

	for (int iter = 0; iter < reps; ++iter) {
		a(iter);
	}

	t2 = omp_get_wtime() - t2;

	std::cout <<"sz = " << SIZE << "time scalar: " << t << " time generic access: " << t2 << std::endl;

	return 0;
}


