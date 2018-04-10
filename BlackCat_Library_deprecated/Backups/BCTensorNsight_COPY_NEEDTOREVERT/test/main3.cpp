#include <BlackCat_Tensors.h>
#include <iostream>
#include <omp.h>
#include "math.h"
#include "time.h"
int tq_test() {
	std::cout << " ehere" << std::endl;
	Tensor_Jack<float> a = { 2, 3 };
	Tensor_Jack<float> b = { 2, 3 };
	Tensor_Jack<float> c = { 2, 3 };
	Tensor_Jack<float> t = { 2, 3 };

	b.randomize(-50, 50);
	c.randomize(-50, 50);

	b.print();
	c.print();

//	t = b + c % c;
	t.print();
	a = Scalar<float>(200.0) - t;

	a.print();

	std::cout << "success " << std::endl;
	return 0;
}

void matrix_tests() {

	tq_test();

	Vector<float> a(5);

	a.randomize(-1, 1);
	a.print();

	a[0].print();

	a[3].print();

	a.print();

	std::cout << std::endl;

	Matrix<float> b(2, 3);
	Matrix<float> c(3, 2);
	Matrix<float> d(2, 2);

	for (int i = 0; i < b.size(); ++i) {
		b.accessor()[i] = i + 1;
		c.accessor()[i] = i + 7;
	}

	d.randomize(-1, 1);

	d.print();

	b.print();
	c.print();

	//d = b * c + b[1][1];

	d.print();

	c.print();

//	c.t().print();
	std::cout << " success " << std::endl;
}
void add(double* a, double* b, double* c, int sz) {
//#pragma omp parallel for

	for (int i = 0; i < sz; ++i) {
		a[i] = b[i] + c[i];
	}

//#pragma omp barrier
}
void add(double* a, double* b, double* c, double* d, int sz) {
//#pragma omp parallel for

	for (int i = 0; i < sz; ++i) {
		a[i] = b[i] + c[i] + d[i];
	}
//#pragma omp barrier
}

#include <pthread.h>
void printCheck() {
	std::cout << pthread_self() << std::endl;
}

int SpeedTests() {

	//omp_set_num_threads(6);

	printCheck();

	Matrix<double> alpha(1000, 1000);
	Matrix<double> beta(1000, 1000);
	Matrix<double> gamma(1000, 1000);

	alpha.randomize(-100, 100);
	beta.randomize(-100, 100);

	float t;
	t = omp_get_wtime();
	printf("Calculating...(optimized)\n");
	for (int i = 0; i < 1000; ++i) {

		gamma = alpha + beta + alpha;
	}

	printf("adding alpha, beta, alpha together (optimized code): %d\n", t);
	t = omp_get_wtime() - t;
	printf("Time: (%f seconds).\n", t, ((float) t));

//
	double* c = gamma.accessor().getData();
	double* a = alpha.accessor().getData();
	double* b = beta.accessor().getData();

	int sz = gamma.size();
	t = omp_get_wtime();

	std::cout << std::endl << std::endl;
	printf("adding alpha, beta, alpha, than alpha again (\"generic dumb version\") \n");

	for (int i = 0; i < 1000; ++i) {
		add(c, a, b, sz);
		add(c, c, a, sz);
	}
	t = omp_get_wtime() - t;
	printf("Time: (%f seconds).\n", t, ((float) t));

	std::cout << "success " << std::endl;

	std::cout << std::endl << std::endl;
	printf("Best possible version (hardcoded)  \n");
	t = omp_get_wtime();

	for (int i = 0; i < 1000; ++i) {
		add(c, a, b, b, sz);
	}
	t = omp_get_wtime() - t;
	printf("Time: (%f seconds).\n", t, ((float) t));

	std::cout << std::endl << std::endl;
	std::cout << "success " << std::endl;
}


template<int size>
struct begin {
	static void init_helper(double* d) {
		d[size] = size;
		std::cout << 1 + size<< std::endl;

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

};

int main() {

	SpeedTests();
//
//	Matrix<double> a(2, 3);
//
//	a.randomize(-3, 3);
//
//	Matrix<double> b(3, 2);
//	b.randomize(-3, 3);
//
//	Matrix<double> c(2, 2);
//
//	a.print();
//	b.print();
//
//	c = a * b;
//
//	Matrix<double> d(2, 3);
//	d.randomize(-1, 1);
////	d = a ;
//	d.print();
//
//	//c.print();
//
//	c = a * d.t();
//
//	c.print();
//	return 0;
}

//
//namespace helper {
//	template<int dim, int ... dims>
//	void generateShape(int* s) {
//		*s = dim;
//		generateShape<dims...> [&s[1]];
//	}
//	template<int dim>
//	void generateShape(int* s) {
//		*s = dim;
//	}
////
//	template<int d>
//	constexpr int sizeof_ary() {
//		return d;
//	}
//
//	template<int d, int ... dims>
//	constexpr int sizeof_ary() {
//		return d * sizeof_ary<dims...>();
//	}
//
//template<class T, int begin, int end>
//void init_help(T* ary){
//	ary[begin] = begin;
//	init_help<T, begin + 1, end>(ary);
//}
//
//template<int a, int b>
//struct max {
//	const int value = a > b ? a : b;
//};
//
//
//template<class T, int sz>
//void init(T* ary) {
//	ary[sz] = sz;
//
//	sz != 0 ? init<T, sz - 1>(ary) : ary[0] = 0;
//}
//
//
//template<class T>
//void init(T* ary, int sz) {
//	for (int i = 0; i < sz; ++i) {
//		ary[i] = i;
//	}
//}
//
//
//template<int numb_elements>
//struct tensor_parent {
//
//	const int size = numb_elements;
//
//	double* ary;
//
//	tensor_parent() {
//		ary = new double[size];
//	}
//};
//
//template<int ... dimensions>
//struct tensor : tensor_parent<90> {
//
//	int* shape;
//
//};
//
//int main() {
//
//	const int sz = 90;
//	tensor<30, 3> alpha;
//	tensor<30, 3> beta;
//
//
////Benchmarking template meta programming
//	std::cout << std::endl << std::endl;
//	printf("adding alpha, beta, alpha, than alpha again (\"generic dumb version\") \n");
//
//	//for (int i = 0; i < 1000; ++i) {
//		init(beta.ary, sz);
//	//}
//	float t;
//	t = omp_get_wtime() - t;
//	printf("Time: (%f seconds).\n", t, ((float) t));
//
//	std::cout << "success " << std::endl;
//
//	std::cout << std::endl << std::endl;
//	printf("Best possible version (hardcoded)  \n");
//	t = omp_get_wtime();
//
//	//for (int i = 0; i < 1000; ++i) {
//		init<double, sz>(alpha.ary);
//	//}
//	t = omp_get_wtime() - t;
//	printf("Time: (%f seconds).\n", t, ((float) t));
//
//	std::cout << std::endl << std::endl;
//	std::cout << "success " << std::endl;
//}}

