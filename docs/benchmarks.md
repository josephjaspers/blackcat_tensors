# Benchmarks 

Compiler: NVCC -O3 (no-openmp)  
Machine: G751jy Asus  
OS: Ubuntu 16.04.  


Note:
The performance columns represent performance difference. (Greater than 1 is better, less than 1 is worse)  
Under perfect benchmark conditions, the performance should almost always be equal to or worse than the 
handcoded-C versions. The fact that BC-times showcase occasional performance increases should be interpreted that the cost of this abstraction has almost near-0 impact on the performance. 

----------------------------------------------------------------------------------------


BC time represents the performance of BC::Vector<float>.   
C-impl and Cuda-impl are 'handcoded' baslines t bench against.   
The timings are the sum of each computation.  

Operation: `a = b + c - d / e`   
Repetitions: 1000  


| Size | C-impl | BC_host | Host-performance | Cuda-impl | BC_device | Device-Performance | 
| ---  | --- | --- | --- | --- | --- | --- | 
|100|6.0109e-05|5.0272e-05|1.19568|0.00234129|0.00229101|1.02194|
|500|0.000258773|0.000243724|1.06175|0.00238337|0.00230816|1.03258|
|1000|0.000536694|0.00048735|1.10125|0.00229887|0.00232279|0.989702|
|2500|0.00124706|0.00121823|1.02366|0.00238673|0.00234499|1.0178|
|5000|0.00249811|0.00243681|1.02516|0.00225909|0.00233199|0.968739|
|75000|0.0379705|0.0367494|1.03323|0.00245717|0.00240522|1.0216|
|10000|0.00505725|0.00498402|1.01469|0.00229609|0.00232528|0.987449|


----------------------------------------------------------------------------------------
Compiler: GCC 8 -O3, -fopenmp  
Machine: k5708d Asus  
OS: Ubuntu 18.04.  

Time represents the total sum of each rep.


BC time represents the performance of BC::Vector<double>.  
The baseline time is performance of the hardcoded implementation.  
  
Operation: `a = b + c - d / e`  
Repetitions: 10

|Size | BC time | Baseline | Performance difference |
| --- | --- | --- | --- |
|1000000|0.316211|0.293648|0.928644|
|1500000|0.368338|0.313048|0.849892|
|2250000|0.403998|0.305348|0.755816|
|3375000|0.447446|0.405975|0.907318|
|5062500|0.440331|0.487939|1.10812|
|7593750|0.512234|0.519952|1.01507|
|11390625|0.685765|0.70168|1.02321|
|17085938|0.893298|0.934586|1.04622|
|25628908|1.32054|1.25206|0.948138|
|38443360|1.84245|1.78575|0.969225|

[source code](https://github.com/josephjaspers/BlackCat_Tensors/blob/master/benchmarks/elementwise.h)

------------------------------------------------------------------------------------------

BC time represents the performance of BC::Matrix<double>.  
The baseline time is performance of the hardcoded implementation.  
Matrices are of dimension size*size. 

Operation: `y = w * x + b`  
Repetitions: 100

|Size | BC time | Baseline | Performance difference |
| --- | --- | --- | --- |
|128|0.00638753|0.00861884|1.34932|
|192|0.0400103|0.0165301|0.413147|
|288|0.0552596|0.0539122|0.975617|
|432|0.197306|0.181367|0.919215|
|648|0.553671|0.554848|1.00213|
|972|1.81755|1.98156|1.09024|
|1458|6.19255|7.3078|1.1801|
|2187|22.1012|21.9434|0.992862|

[source code](https://github.com/josephjaspers/BlackCat_Tensors/blob/master/benchmarks/benchmark_matmul_reordering.h)


