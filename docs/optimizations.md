# Optimizations:

BlackCat_Tensor's utilize's two optimizations to improve its performance; expression-templates, and compile-time expression reordering. 


#### Expression-Templates
The standard expression-template system allows converting multiple element-wise operations into single for loops.
    For example an operation such as:
    y = a + b - c / d; 

Will automatically produce assembly code that is identical to

```cpp
for (int i = 0; i < y.size(); ++i)
    y[i] = a[i] + b[i] - c[i] / d[i];
```

This in short allows users to use code that appears to be identical to mathematical expressions with performance equivalent to hand written code.

If you are interested in how expression templates are implemented, (Todd Veldhuizen is the original creator of expression-templates)
        https://www.cct.lsu.edu/~hkaiser/spring_2012/files/ExpressionTemplates-ToddVeldhuizen.pdf        



#### Compile-Time Expression Reordering
BlackCat_Tensor's now supports an injection system for gemm (and other BLAS routines) to reduce the generation of temporaries. 
Operations utilizing matrix multiplication in which possible "injections" are available will not use a temporary to store the result of a matrix-mul operation.

This system detects complicated matrix-mul problems such as forward-propagation in NeuralNetworks.

```cpp
y = sigmoid(w * x + b);     //forward propagation
```

The "optimized" hard-coded form would be 

1) evalute the product of `w` and `x` to `y`.
`y = w * x` (which is evaluated via a single BLAS call gemm(y, w, x))

2) evaluate the element-wise operation of
`y = sigmoid (y + b)` (which is evaluated with a single for loop through the expression-template system)

BlackCat_Tensors at compile detects that the reordering above is possible and executes the expression as such.

***Caveat***

BCT cannot detect if an alias is used.
so problems such as:
     y = y * x
will cause the BLAS_gemm call to start writing to Y as it is still calculating the product. 
When reusing tensors you may use aliases to skip this optimization.
Example: y.alias() = y * x
