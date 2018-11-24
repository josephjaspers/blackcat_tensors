# Aliases 

BlackCat_Tensor's defines a numerous amount of user-types.

Two template arguments are commonly supplied, scalar_t, (scalar type) and allocator_t (how to allocate the memory). 

#### The Core Tensor Types: 
```cpp  
    Tensor<int x, scalar_t, allocator_t> 
    Scalar<scalar_t, allocator_t>
    Vector<scalar_t, allocator_t>
    Matrix<scalar_t, allocator_t>
    Cube<scalar_t, allocator_t>
```

#### View Types:
- Accepts a Tensor to construct. Shares the internal pointer and is not mutable. 
- Construction is a no-copy operation, though the data is not modifiable. 
```cpp
    Tensor_View<int x, scalar_t, allocator_t>
    Scalar_View<scalar_t, allocator_t>
    Vector_View<scalar_t, allocator_t>
    Matrix_View<scalar_t, allocator_t>
    Cube_View<scalar_t, allocator_t>
```
#### Shared Types:
- Accepts a non-const Tensor to construct. Shares the internal_pointer but is mutable
```cpp
    Tensor_Shared<int x, scalar_t, allocator_t> 
    Scalar_Shared<scalar_t, allocator_t> 
    Vector_Shared<scalar_t, allocator_t>
    Matrix_Shared<scalar_t, allocator_t>
    Cube_Shared<scalar_t, allocator_t>
```
#### Expression Types:
- Any tensor that is expression (non-memory owning) or Tensor (memory owning) of the apropriate dimensionality.
- Cannot be constructed directly, used only for method parameters. 
```cpp
    expr::Tensor<int> 
    expr::Scalar
    expr::Vector
    expr::Matrix
    expr::Cube
  ```
