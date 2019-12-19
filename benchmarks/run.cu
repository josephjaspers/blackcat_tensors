#include "../blackcat/tensors.h"
#include "benchmark_suite.h"
#include "benchmark_matmul_reordering.h"
#include "elementwise.h"

int main() {
//	BC::print("matmul suite");
//	benchmark_matmul_suite();

	BC::print("for loop suite");
	benchmark_forloop_suite();
}
