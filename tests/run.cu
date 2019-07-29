#include "run.h"
#include <iostream> 

#include <cuda.h> 
int main() {

	int driver_version;
	cudaDriverGetVersion(&driver_version);
	int runtime_version;
	cudaRuntimeGetVersion(&runtime_version);
	std::cout << "Runtime version: " << runtime_version  << std::endl;
	std::cout << "Driver version: " << driver_version << std::endl; 
	BC::tests::run();
}
