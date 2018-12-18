# Methods

`scalar_t` is used to denote the scalar-type of the tensor.  
`allocator_t` is used to denote the allocator-template argument.  
`expression_t` is used to represent any non-evaluated mathematical expression.   
Note: Many of the return types have been abreviated. The underlying implementation of these is not relevant to the user. 

#### Data Access 
| static | return type | method name | parameters | const/non-const | documentation | alias-methods |
| --- | --- | --- | --- | --- | --- | --- |
| --- | slice | operator[] | BC::size_t  | both | Returns a slice of the tensor. IE Cube returns a matrix slice, Matrix returns a column, etc | slice |
| --- | scalar_obj | operator() | BC::size_t  | both | Returns a scalar object. Access to this data is safe. | scalar | 
| --- | vector | diag | int=0 | both | Returns the diagnol of a matrix. A positive integer will have the diagnol start from the top left corner. A negative integer will have the diagnol end n from the bottom right |
| --- | slice | col | BC::size_t  | both | Returns a column of a matrix. |
| --- | transpose_view | transpose | --- | both | returns a transpose view of a Matrix or Vector. Cannot transpose in place. Matrix = Matrix.transpose() is undefined. | t |
| --- | vector | row | BC::size_t  | both | returns a row of a matrix. |
| --- | view | transpose | BC::size_t  | both | returns a row of a matrix. |
| X | reshape | reshape | tensor, and ints... | both | returns a reshaped view of the tensor parameter. Does not modify the original tensor |
| X | chunk | chunk | tensor, and ints... | both | returns a reshaped view of the tensor parameter. Does not modify the original tensor |

#### Iterator Methods
| static | return type | method name | parameters | const/non-const | documentation | alias-methods |
| --- | --- | --- | --- | --- | --- | --- |
| --- | cw_iterator | begin | --- | both | Returns the begining of a coefficientwise iterator. | --- |
| --- | cw_iterator | end | --- | both | Returns the end of a coefficientwise iterator. | --- |
| --- | cw_iterator | cbegin | --- | const | Explcit const version of begin. | --- |
| --- | cw_iterator | cend | --- | const | Explcit const version of end. | --- |
| --- | cw_iterator | rbegin | --- | both | Returns the begining of a coefficientwise reverse iterator. | --- |
| --- | cw_iterator | rend | --- | both | Returns the begining of a coefficientwise reverse iterator. | --- |
| --- | cw_iterator | crbegin | --- | const | Explicit const version of rbegin. | --- |
| --- | cw_iterator | crend | --- | const | Explicit const version of rend. | --- |
| --- | nd_iterator | nd_begin | --- | both | Returns the begining of a multidimensional iterator (iterates along outer stride). | --- |
| --- | nd_iterator | nd_end | --- | both | Returns the end of a multidimensional iterator. | --- |
| --- | nd_iterator | nd_cbegin | --- | const | Explicit const version of nd_begin. | --- |
| --- | nd_iterator | nd_cend | --- | const | Explicit const version of nd_end. | --- |
| --- | nd_iterator | nd_rbegin | --- | both | Returns the begining of a multidimensional reverse iterator. (Iterates along outer stride). | --- |
| --- | nd_iterator | nd_rend | --- | both | Returns the end of a multidimensional reverse iterator. | --- |
| --- | nd_iterator | nd_crbegin | --- | const | Explicit const version of nd_rbegin. | --- |
| --- | nd_iterator | nd_crend | --- | const | Explicit const version of nd_rend. | --- |
| --- | cw_iterator | iter | int=0, int=size | both | Returns an iterator proxy, used for range-convienance. | --- |
| --- | cw_iterator | nd_iter | int=0, int=size | both | Returns an iterator proxy, used for range-convienance. | --- |
| --- | cw_iterator | reverse_iter | int=0, int=size | both | Returns an iterator proxy, used for range-convienance. | --- |
| --- | cw_iterator | nd_reverse_iter | int=0, int=size | both | Returns an iterator proxy, used for range-convienance. | --- |

#### Operations
| static | return type | method name | parameters | const/non-const | documentation | alias-methods |
| --- | --- | --- | --- | --- | --- | --- |
| --- | tensor& | += | tensor or scalar | non-const | --- | --- |
| --- | tensor& | -= | tensor or scalar | non-const | --- | --- |
| --- | tensor& | %= | tensor or scalar | non-const | Element-wise multiplication (to differentiate from matrix multiplication) | --- |
| --- | tensor& | /= | tensor or scalar | non-const | --- | --- |
| --- | expression_t | + | tensor or scalar | both | --- | --- |
| --- | expression_t | - | tensor or scalar | both | --- | --- |
| --- | expression_t | % | tensor or scalar | both | Element-wise multiplication (to differentiate from matrix multiplication) | --- |
| --- | expression_t | / | tensor or scalar | both | --- | --- |
| --- | expression_t | == | tensor or scalar | both | --- | --- |
| --- | expression_t | > | tensor or scalar | both | --- | --- |
| --- | expression_t | < | tensor or scalar | both | --- | --- |
| --- | expression_t | >= | tensor or scalar | both | --- | --- |
| --- | expression_t | <= | tensor or scalar | both | --- | --- |
| --- | expression_t | * | tensor or scalar | both | Executes one of the following BLAS calls gemm, gemv, ger, dot, or scalarmul depending upon the dimensionality of the parameters. This is detected at compile-time, and does not incur any branching | --- |
| --- | expression_t | - | --- | both | Negation of a tensor. | --- |
| --- | expression_t | un_expr | functor | Returns a user-defined unary_expression object that will be laziliy evaluated. | --- |
| --- | expression_t | bi_expr | functor, tensor or scalar | Returns a user-defined binary_expression object that will be laziliy evaluated. | --- |

#### Functions 
| static | return type | method name | parameters | const/non-const | documentation | alias-methods |
| --- | --- | --- | --- | --- | --- | --- |
| --- | scalar_t | min | --- | both | --- | --- |
| --- | scalar_t | max | --- | both | --- | --- |
| --- | void | rand | --- | --- | non-const | randomize |
| --- | functor | for_each | functor | both | Convenient-definition of for_each. Identical to BC::for_each(tensor.begin(), tensor.end(), functor) | --- |
| --- | void | sort | --- | non-const | Implemenation is dependent upon gpu vs cpu allocation and std and thrust's implementation. | --- |



#### Utility
| static | return type | method name | parameters | const/non-const | documentation | alias-methods |
| --- | --- | --- | --- | --- | --- | --- |
| --- | void | print | --- | both | Formatted print to console. | --- |
| --- | void | print_sparse | --- | both | Formatted print to console, ignoring 0's. | --- |
| --- | void | print_dimensions | --- | both | Output dimensions of a tensor. | --- |
| --- | void | print_leading_dimensions | --- | both | Output outer dimensions of a tensor (strides). | --- |
| --- | void | print_block_dimensions | --- | both | Output the block_dimensions of a tensor (IE a 3x4 matrix will output. `[3][12]`) | --- |

#### CMath
The following Cmath functions are supported through the `BC` namespace. These expressions will automatically be scalarized. (lazy evaluated)
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
```
#### Other functions
```
logistic              1 / (1 + exp(-x))
dx_logistic,          x * (1 - logistic(x))
cached_dx_logistic    x * (1 - x;
dx_tanh               1 - pow(tanh(x, 2))
cached_dx_tanh        1 - pow(x, 2)
relu                  max(0, x)
dx_relu               x > 0 ? 1 : 0
cached_dx_relu        x > 0 ? 1 : 0
```
#### Algorithms
Algorithms are not implemented by BCT. They are forwarded to the standard library or thrust implementation. 
The purpose of BC::Algorithms is to automatically forward to the correct implementation by detecting how the memory of a tensor was allocated at compile time.
```
for_each
count
count_if
find
find_if
find_if_not
copy
copy_if
copy_n
fill
fill_n
transform
generate
generate_n
replace
replace_if
replace_copy
replace_copy_if
swap
swap_ranges
reverse
reverse_copy
stable_sort
max_element
min_element
minmax_element
equal
```
