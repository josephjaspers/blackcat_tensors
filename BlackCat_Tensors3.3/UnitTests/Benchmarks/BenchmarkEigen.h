#ifndef BC_EIGEN_BENCH
#define BC_EIGEN_BENCH

#include <Eigen/Dense>
#include "../../BlackCat_Tensors.h"
#include <iostream>
#include <omp.h>
#include "time.h"
#include <omp.h>
#include <vector>

using BC::Matrix;
using Eigen::MatrixXd;
namespace BC_EIGEN_BENCHMARK {

std::string benchmark1_str() {
	return "a = b + c * 3 * d - e ";
}

template<int SIZE, int repetitions>
float benchmark1() {

	BC::Matrix<double, BC::CPU> bc_a(SIZE,SIZE);
	BC::Matrix<double, BC::CPU> bc_b(SIZE,SIZE);
	BC::Matrix<double, BC::CPU> bc_c(SIZE,SIZE);
	BC::Matrix<double, BC::CPU> bc_d(SIZE,SIZE);
	BC::Matrix<double, BC::CPU> bc_e(SIZE,SIZE);
	BC::Scalar<double, BC::CPU> scal(3);

	Eigen::MatrixXd eg_a(SIZE,SIZE);
	Eigen::MatrixXd eg_b(SIZE,SIZE);
	Eigen::MatrixXd eg_c(SIZE,SIZE);
	Eigen::MatrixXd eg_d(SIZE,SIZE);
	Eigen::MatrixXd eg_e(SIZE,SIZE);

	bc_a.randomize(0, 10000);
	bc_b.randomize(0, 10000);
	bc_c.randomize(0, 10000);
	bc_d.randomize(0, 10000);
	bc_e.randomize(0, 10000);

	//copy to ensure same parameters
	for (int i = 0; i < SIZE * SIZE; ++i) {
		eg_a(i) = bc_a.data().getIterator()[i];
		eg_b(i) = bc_b.data().getIterator()[i];
		eg_c(i) = bc_c.data().getIterator()[i];
		eg_d(i) = bc_d.data().getIterator()[i];
		eg_e(i) = bc_e.data().getIterator()[i];
	}

	float eigen_time = omp_get_wtime();

		for (int i = 0; i < repetitions; ++i) {
			eg_a = eg_b - eg_c * 3 * (eg_d - eg_e);
		}

		eigen_time = omp_get_wtime() - eigen_time;


	float blackcat_time = omp_get_wtime();

	for (int i = 0; i < repetitions; ++i) {
		bc_a = bc_b + bc_c * scal * bc_d - bc_e;
	}
	blackcat_time = omp_get_wtime() - blackcat_time;



	float time_dif = (blackcat_time - eigen_time);
	std::string winner = time_dif < 0 ? ("Blackcat_Tensors better_by " + std::to_string(time_dif * -1))
											: ("Eigen better by " + std::to_string(time_dif));

	std::string info1 = "SIZE = [" + std::to_string(SIZE) + "][" + std::to_string(SIZE) + "]";
	std::string info2 = " Reps = " + std::to_string(repetitions);
	std::string bc_time = "BLACKCAT_TENSORS_TIME:  " + std::to_string(blackcat_time);
	std::string eg_time = "EIGEN TIME: " + std::to_string(eigen_time);

	struct padder {
		static void pad_string(int sz, std::string& str) {
			if (str.length() < sz) {
				str += std::string(sz - str.length(), ' ');
			}
		}
	};

	padder::pad_string(20, info1);
	padder::pad_string(20, info2);
	padder::pad_string(40, bc_time);
	padder::pad_string(40, eg_time);


	std::cout<<  info1 << info2 << bc_time << eg_time << winner << std::endl;


	return 0;
}


std::string benchmark2_str() {
	return "a = b + c + d + e";
}

template<int SIZE, int repetitions>
float benchmark2() {
	const int reps = repetitions;


	BC::Matrix<double, BC::CPU> bc_a(SIZE,SIZE);
	BC::Matrix<double, BC::CPU> bc_b(SIZE,SIZE);
	BC::Matrix<double, BC::CPU> bc_c(SIZE,SIZE);
	BC::Matrix<double, BC::CPU> bc_d(SIZE,SIZE);
	BC::Matrix<double, BC::CPU> bc_e(SIZE,SIZE);
	BC::Scalar<double, BC::CPU> scal(3);

	Eigen::MatrixXd eg_a(SIZE,SIZE);
	Eigen::MatrixXd eg_b(SIZE,SIZE);
	Eigen::MatrixXd eg_c(SIZE,SIZE);
	Eigen::MatrixXd eg_d(SIZE,SIZE);
	Eigen::MatrixXd eg_e(SIZE,SIZE);

	bc_a.randomize(0, 10000);
	bc_b.randomize(0, 10000);
	bc_c.randomize(0, 10000);
	bc_d.randomize(0, 10000);
	bc_e.randomize(0, 10000);

	//copy to ensure same parameters
	for (int i = 0; i < SIZE * SIZE; ++i) {
		eg_a(i) = bc_a.data().getIterator()[i];
		eg_b(i) = bc_b.data().getIterator()[i];
		eg_c(i) = bc_c.data().getIterator()[i];
		eg_d(i) = bc_d.data().getIterator()[i];
		eg_e(i) = bc_e.data().getIterator()[i];
	}

	float eigen_time = omp_get_wtime();

		for (int i = 0; i < reps; ++i) {
			eg_a = eg_b + eg_c + eg_d + eg_e;
		}

		eigen_time = omp_get_wtime() - eigen_time;


	float blackcat_time = omp_get_wtime();

	for (int i = 0; i < reps; ++i) {
		bc_a = bc_b + bc_c + bc_d + bc_e;
	}
	blackcat_time = omp_get_wtime() - blackcat_time;



	float time_dif = (blackcat_time - eigen_time);
	std::string winner = time_dif < 0 ? ("Blackcat_Tensors better_by " + std::to_string(time_dif * -1))
											: ("Eigen better by " + std::to_string(time_dif));

	std::string info1 = "SIZE = [" + std::to_string(SIZE) + "][" + std::to_string(SIZE) + "]";
	std::string info2 = " Reps = " + std::to_string(repetitions);
	std::string bc_time = "BLACKCAT_TENSORS_TIME:  " + std::to_string(blackcat_time);
	std::string eg_time = "EIGEN TIME: " + std::to_string(eigen_time);

	struct padder {
		static void pad_string(int sz, std::string& str) {
			if (str.length() < sz) {
				str += std::string(sz - str.length(), ' ');
			}
		}
	};

	padder::pad_string(20, info1);
	padder::pad_string(20, info2);
	padder::pad_string(40, bc_time);
	padder::pad_string(40, eg_time);


	std::cout<<  info1 << info2 << bc_time << eg_time << winner << std::endl;


	return 0;
}

std::string benchmark3_str() {
	return "a = b.convolve(c.convolve(d))";
}

template<int SIZE, int repetitions>
float benchmark3() {
	const int reps = repetitions;


	BC::Matrix<double, BC::CPU> bc_a(SIZE-4,SIZE-4);
	BC::Matrix<double, BC::CPU> bc_b(3,3);
	BC::Matrix<double, BC::CPU> bc_c(SIZE,SIZE);
	BC::Matrix<double, BC::CPU> bc_d(SIZE-2,SIZE-2);

	bc_a.randomize(0, 10000);
	bc_b.randomize(0, 10000);
	bc_c.randomize(0, 10000);
	bc_d.randomize(0, 10000);


	float NAIVE_TIME = omp_get_wtime();

		for (int i = 0; i < reps; ++i) {
			bc_d = bc_b.x_corr<2>(bc_c);
			bc_a = bc_b.x_corr<2>(bc_d);
		}

		NAIVE_TIME = omp_get_wtime() - NAIVE_TIME;


	float NESTED_TIME = omp_get_wtime();

	for (int i = 0; i < reps; ++i) {
		bc_a = bc_b.x_corr<2>(bc_b.x_corr<2>(bc_c));
	}
	NESTED_TIME = omp_get_wtime() - NESTED_TIME;



	float time_dif = (NESTED_TIME - NAIVE_TIME);
	std::string winner = time_dif > 0 ? ("Blackcat_Tensors Naive better_by " + std::to_string(time_dif * -1))
											: ("BlackCat Nested better by " + std::to_string(time_dif));

	std::string info1 = "SIZE = [" + std::to_string(SIZE) + "][" + std::to_string(SIZE) + "]";
	std::string info2 = " Reps = " + std::to_string(repetitions);
	std::string bc_time = "BLACKCAT_Naive Conv:  " + std::to_string(NAIVE_TIME);
	std::string eg_time = "BLACKCAT_Nested Conv: " + std::to_string(NESTED_TIME);


	struct padder {
		static void pad_string(int sz, std::string& str) {
			if (str.length() < sz) {
				str += std::string(sz - str.length(), ' ');
			}
		}
	};

	padder::pad_string(20, info1);
	padder::pad_string(20, info2);
	padder::pad_string(40, bc_time);
	padder::pad_string(40, eg_time);


	std::cout<<  info1 << info2 << bc_time << eg_time << winner << std::endl;


	return 0;
}


template<int SIZE, int repetitions, int krnl_size =3>
float benchmark4() {

	const int reps = repetitions;
	const int K = krnl_size - 1;


	BC::Matrix<double, BC::CPU> bc_a(SIZE - K * 3,SIZE  - K * 3);
	BC::Matrix<double, BC::CPU> bc_b(krnl_size,krnl_size);
	BC::Matrix<double, BC::CPU> bc_c(SIZE,SIZE);
	BC::Matrix<double, BC::CPU> bc_d(SIZE - K * 2,SIZE - K * 2);
	BC::Matrix<double, BC::CPU> bc_e(SIZE-  K, SIZE- K);

	bc_a.randomize(0, 10000);
	bc_b.randomize(0, 10000);
	bc_c.randomize(0, 10000);
	bc_d.randomize(0, 10000);


	float NAIVE_TIME = omp_get_wtime();

		for (int i = 0; i < reps; ++i) {
			bc_e = bc_b.x_corr<2>(bc_c);
			bc_d = bc_b.x_corr<2>(bc_e);
			bc_a = bc_b.x_corr<2>(bc_d);
		}

		NAIVE_TIME = omp_get_wtime() - NAIVE_TIME;


	float NESTED_TIME = omp_get_wtime();

	for (int i = 0; i < reps; ++i) {
		bc_a = bc_b.x_corr<2>(bc_b.x_corr<2>(bc_b.x_corr(bc_c)));
	}
	NESTED_TIME = omp_get_wtime() - NESTED_TIME;



	float time_dif = (NESTED_TIME - NAIVE_TIME);
	std::string winner = time_dif > 0 ? ("Blackcat_Tensors Naive better_by " + std::to_string(time_dif * -1))
											: ("BlackCat Nested better by " + std::to_string(time_dif));

	std::string info1 = "SIZE = [" + std::to_string(SIZE) + "][" + std::to_string(SIZE) + "]";
	std::string info2 = " Reps = " + std::to_string(repetitions);
	std::string bc_time = "BLACKCAT_Naive Conv:  " + std::to_string(NAIVE_TIME);
	std::string eg_time = "BLACKCAT_Nested Conv: " + std::to_string(NESTED_TIME);

	struct padder {
		static void pad_string(int sz, std::string& str) {
			if (str.length() < sz) {
				str += std::string(sz - str.length(), ' ');
			}
		}
	};

	padder::pad_string(20, info1);
	padder::pad_string(20, info2);
	padder::pad_string(40, bc_time);
	padder::pad_string(40, eg_time);


	std::cout<<  info1 << info2 << bc_time << eg_time << winner << std::endl;


	return 0;
}
}
#endif
//
//int main() {
//
//	std::cout << "BENCHMARKING - 03 OPTIMIZATIONS, NO OPENMP" << std::endl;
//	std::cout << "Benchmarking: " << BC_EIGEN_BENCHMARK::benchmark1_str() << std::endl;
//
//	BC_EIGEN_BENCHMARK::benchmark1<4,     100000>();
//	BC_EIGEN_BENCHMARK::benchmark1<8,     100000>();
//	BC_EIGEN_BENCHMARK::benchmark1<16,    10000>();
//	BC_EIGEN_BENCHMARK::benchmark1<64,    10000>();
//	BC_EIGEN_BENCHMARK::benchmark1<128,   1000>();
//	BC_EIGEN_BENCHMARK::benchmark1<256,   1000>();
//	BC_EIGEN_BENCHMARK::benchmark1<512,   100>();
//
//	std::cout << "Benchmarking: " << BC_EIGEN_BENCHMARK::benchmark2_str() << std::endl;
//
//	BC_EIGEN_BENCHMARK::benchmark2<4,     100000>();
//	BC_EIGEN_BENCHMARK::benchmark2<8,     100000>();
//	BC_EIGEN_BENCHMARK::benchmark2<16,    10000>();
//	BC_EIGEN_BENCHMARK::benchmark2<64,    10000>();
//	BC_EIGEN_BENCHMARK::benchmark2<128,   1000>();
//	BC_EIGEN_BENCHMARK::benchmark2<256,   1000>();
//	BC_EIGEN_BENCHMARK::benchmark2<512,   100>();
//
//	std::cout << " success  main" << std::endl;
//
//	return 0;
//}

