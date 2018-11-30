# Benchmarks 

Compiler: GCC 8 -O3, -fopenmp  
Machine: k5708d Asus  
OS: Ubuntu 18.04.  

----------------------------------------------------------------------------------------
Time represents the total sum of each rep.

Operation: `a = b + c - d / e`  
Repetitions: 10

|Size | BC time | Baseline time | Performance difference |
| --- | --- | --- | --- |
|1000000|0.361085|0.372586|0.96913|
|1500000|0.646324|0.6108|1.05816|
|2250000|0.919561|0.900934|1.02067|
|3375000|1.36323|1.45519|0.936804|
|5062500|2.03941|2.15926|0.944498|
|7593750|3.13589|3.15575|0.993706|
|11390625|4.69236|4.64957|1.0092|
|17085938|6.70392|6.68943|1.00217|
|25628908|9.98712|10.0885|0.989948|
|38443360|14.765|14.7707|0.999619|

[source code](https://github.com/josephjaspers/BlackCat_Tensors/blob/master/benchmarks/elementwise.h)

