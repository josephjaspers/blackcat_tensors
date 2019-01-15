# Aliases 

BlackCat_Tensor's defines a numerous amount of user-types.

Two template arguments are commonly supplied, value_type, (scalar type) and allocator_t (how to allocate the memory). 

#### The Core Tensor Types: 
```cpp  
    Tensor<int x, value_type, allocator_t> 
    Scalar<value_type, allocator_t>
    Vector<value_type, allocator_t>
    Matrix<value_type, allocator_t>
    Cube<value_type, allocator_t>
```

#### View Types:
- Accepts a Tensor to construct. Shares the internal pointer and is not mutable. 
- Construction is a no-copy operation, though the data is not modifiable. 
```cpp
    Tensor_View<int x, value_type, allocator_t>
    Scalar_View<value_type, allocator_t>
    Vector_View<value_type, allocator_t>
    Matrix_View<value_type, allocator_t>
    Cube_View<value_type, allocator_t>

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
