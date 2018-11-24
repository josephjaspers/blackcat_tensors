# CMath Functions

BlackCat_Tensor's support a large quantity of the cmath functions. 
All element-wise functions are automatically scalarized when written in nested expressions. 

```cpp

BC::Matrix<float> a(m,n);
BC::Matrix<float> b(m,n);
BC::Matrix<float> c(m,n);

a =  b + BC::sin(c); //will lazily evaluate to become a single for-loop. See: [expression_templates](https://github.com/josephjaspers/BlackCat_Tensors/blob/master/docs/algorithms.md)
```

##### Supported Functions
```
abs
acos
acosh
sin
asin
asinh
atan
atanh
cbrt
ceil
cos
cosh
exp
exp2
fabs
floor
fma
isinf
isnan
log
log2
lrint
lround
modf
sqrt
tan
tanh

logistic              1 / (1 + exp(-x))
dx_logistic,          x * (1 - logistic(x))
cached_dx_logistic    x * (1 - x;
dx_tanh               1 - pow(tanh(x, 2))
cached_dx_tanh        1 - pow(x, 2)
relu                  max(0, x)
dx_relu               x > 0 ? 1 : 0
cached_dx_relu        x > 0 ? 1 : 0
```
