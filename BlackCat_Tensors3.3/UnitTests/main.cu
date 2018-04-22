#include <iostream>
#include "../BlackCat_Tensors.h"
#include <type_traits>
#include "SpeedTests.h"
#include <omp.h>

using BC::Vector;
using BC::Matrix;
using BC::Scalar;
using BC::Cube;

//using ml = BC::CPU;
using ml = BC::GPU;

using vec = Vector<float, ml>;
using mat = Matrix<float, ml>;
using scal = Scalar<float, ml>;
using cube = Cube<float, ml>;
using tensor4 = BC::Tensor4<float, ml>;
using tesnor5 = BC::Tensor5<float, ml>;

auto test() {

	mat a(3, 2);
	mat b(2 ,3);
	mat d(2, 3);
	mat e(3, 2);
	mat c(2, 2);
	mat abs(10,10);

	abs.printDimensions();

	std::cout << " param " << std::endl;

	a.printDimensions();

	for (int i = 0; i < 6; ++i)  {
		a(i) = i + 7;
		b(i) = i + 1;
	}


	a.print();
	b.print();

	d = a.t();
	e = b.t();
//
	d.print();
	e.print();

	std::cout << " E" << std::endl;
	e.print();

	std::cout << " simple dot product [following should all have same value]" << std::endl;
	c = d * e;
	c.print();
	std::cout << " dot product -- transpose, transpose" << std::endl;
	c = a.t() * b.t();
	c.print();
	std::cout << " dot product -- transpose, regular" << std::endl;
	c = a.t() * e;
	c.print();
	std::cout << " dot product -- regular, transpose" << std::endl;
	c = d * b.t();
	c.print();
	std::cout << " dot product -- scalar, regular" << std::endl;
	c = d * scal(2.0f) * e + c;
	c.print();
//	std::cout << " dot product -- scalar, regular" << std::endl;
	c = scal(2.0f) * d * e;
	c.print();
	std::cout << " dot product -- regular,  scalar " << std::endl;
//
	c = d * e * scal(2.0f); ////This is the only version that is not accounted for (it is also the least common notation)
	c.print();

	scal A(2.0f);
	scal B(2.0f);

	std::cout << " dot product -- trans, trans " << std::endl;
	c = a.t() * b.t();
	c.print();

	std::cout << " dot product -- trans scalar, trans " << std::endl;
	A.print();
	c = a.t() * A * b.t();
	c.print();
//
	std::cout << " dot product -- scalar trans, trans " << std::endl;
	A.print();
	c = A * a.t() * b.t();
	c.print();

	std::cout << " dot product -- trans, trans scalar " << std::endl;
	A.print();
	c = a.t() * (b.t() * A);
	c.print();

	std::cout << " dot product -- trans scalar, trans scalar " << std::endl;
	A.print();
	c = a.t() * A * (b.t() * A);
	c.print();


	std::cout << " PRINTING CUBE EXAMPLE FORMAT " << std::endl;
	cube cu(2,3, 4);
	cu.zero();
	cu.print();

//
//
	std::cout << " sub vector test " << std::endl;
//
	d.print();
	std::cout << " adding d[1] = d[1] + d[1]" << std::endl;
	d + d;
	d[1] = d[1] + d[1];
	d.print();
	std::cout << " adding d[1][1] = d[1][1] + d[1][1] " << std::endl;


	d[1][1] = d[1][1] + d[1][1];
	d.print();


//	d.row(0).print();

	std::cout << "d print " << d.row(0).size() << std::endl;

	d.col(1).print();

	std::cout << " trying to write " <<std::endl;
	std::ofstream os("save.txt");
	d.write(os);
	os.close();


	std::cout << " trying to read" << std::endl;
	std::ifstream is("save.txt");

	mat readM(d.size());
	readM.read(is);
	readM.print();
	is.close();

	std::cout << readM.data()(1,1) << std::endl;

	std::cout << " general test fin " << std::endl;

}
#include "Benchmarks/BenchmarkEigen.h"

int main() {

test();
std::cout << " post test" << std::endl;

mat krnl(2,2);
krnl.zero();
krnl[0][0] = 1;
krnl[1][1] = 1;

mat img(5,5); img.zero();
for (int m = 0; m < img.rows(); ++m) {
	for (int n = 0; n < img.cols(); ++n)
		if (m ==n)
			img[m][n] = 1;
}

krnl.print();
img.print();

mat m = krnl.x_corr<2>(krnl.x_corr<2>(krnl.x_corr<2>(img)));
m.print();


mat m2 = krnl.x_corr_padded(img);
m2.print();

//auto cor = krnl.x_corr<2>(img);
//auto cor2 = krnl.x_corr<2>(cor);
//mat m = cor2;
//m.print();

	std::cout << "BENCHMARKING - 03 OPTIMIZATIONS" << std::endl;
//	std::cout << "Benchmarking: " << BC_EIGEN_BENCHMARK::benchmark1_str() << std::endl;
////	omp_set_num_threads(2);
//	BC_EIGEN_BENCHMARK::benchmark1<4,     100000>();
//	BC_EIGEN_BENCHMARK::benchmark1<8,     100000>();
//	BC_EIGEN_BENCHMARK::benchmark1<16,    10000>();
//	BC_EIGEN_BENCHMARK::benchmark1<64,    10000>();
//	BC_EIGEN_BENCHMARK::benchmark1<128,   1000>();
//	BC_EIGEN_BENCHMARK::benchmark1<256,   1000>();
//	BC_EIGEN_BENCHMARK::benchmark1<512,   100>();
//
//	std::cout << "Benchmarking: " << BC_EIGEN_BENCHMARK::benchmark2_str() << std::endl;
////
//	BC_EIGEN_BENCHMARK::benchmark2<4,     100000>();
//	BC_EIGEN_BENCHMARK::benchmark2<8,     100000>();
//	BC_EIGEN_BENCHMARK::benchmark2<16,    10000>();
//	BC_EIGEN_BENCHMARK::benchmark2<64,    10000>();
//	BC_EIGEN_BENCHMARK::benchmark2<128,   1000>();
//	BC_EIGEN_BENCHMARK::benchmark2<256,   1000>();
//	BC_EIGEN_BENCHMARK::benchmark2<512,   100>();
//	BC_EIGEN_BENCHMARK::benchmark2<1024,   100>();

//		BC_EIGEN_BENCHMARK::benchmark3<4,     100000>();
//		BC_EIGEN_BENCHMARK::benchmark3<8,     100000>();
//		BC_EIGEN_BENCHMARK::benchmark3<16,    10000>();
//		BC_EIGEN_BENCHMARK::benchmark3<64,    10000>();
//		BC_EIGEN_BENCHMARK::benchmark3<128,   1000>();
//		BC_EIGEN_BENCHMARK::benchmark3<256,   1000>();
//		BC_EIGEN_BENCHMARK::benchmark3<512,   100>();
//		BC_EIGEN_BENCHMARK::benchmark3<1024,   100>();

//		BC_EIGEN_BENCHMARK::benchmark4<8,     100, 4>();
//		BC_EIGEN_BENCHMARK::benchmark4<16,    100, 4>();
//		BC_EIGEN_BENCHMARK::benchmark4<64,    100, 4>();
//		BC_EIGEN_BENCHMARK::benchmark4<128,   100, 4>();
//		BC_EIGEN_BENCHMARK::benchmark4<256,   100, 4>();
//		BC_EIGEN_BENCHMARK::benchmark4<512,   100, 4>();
//		BC_EIGEN_BENCHMARK::benchmark4<1024,   100, 4>();

	//////
	std::cout << " success  main" << std::endl;
//
//	return 0;
}
