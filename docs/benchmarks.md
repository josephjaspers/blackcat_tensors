# Benchmarks 

Compiler: NVCC -O3 (no-openmp)
Machine: G751jy Asus  
OS: Ubuntu 16.04.  


BC time represents the performance of BC::Vector<float>.  
C-impl and Cuda-impl are 'handcoded' baslines t bench against. 
The timings are the sum of each computation. 
Operation: `a = b + c - d / e`  
Repetitions: 1000


| Size | C-impl | BC_host | Host-performance | Cuda-impl | BC_device | Device-Performance | 
| ---  | --- | --- | --- | --- | --- | --- | 
|100|6.0399e-05|5.0273e-05|0.832348|0.002082|0.00211177|1.01429|
|500|0.000270116|0.000250762|0.928349|0.00211264|0.00214798|1.01673|
|1000|0.000537945|0.000501332|0.931939|0.0020678|0.00204336|0.988179|
|2500|0.00124771|0.00121844|0.976538|0.00210658|0.00206911|0.982214|
|5000|0.00251752|0.00243656|0.967841|0.00222548|0.00204068|0.916959|
|75000|0.0385873|0.0376435|0.975543|0.00216992|0.00228764|1.05425|
|10000|0.0055627|0.00515221|0.926205|0.00210182|0.00244626|1.16387|


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


