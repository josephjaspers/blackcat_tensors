# Aliases 

BlackCat_Tensor's defines a numerous amount of user-types.

Two template arguments are commonly supplied, value_type, (scalar type) and allocator_t (how to allocate the memory). 

#### The Core Tensor Types: 
```cpp  
    Tensor<int x, ValueType, Allocator> 
    Scalar<ValueType, Allocator>
    Vector<ValueType, Allocator>
    Matrix<ValueType, Allocator>
    Cube<ValueType, Allocator>
```

#### View Types:
- Accepts a Tensor to construct. Shares the internal pointer and is not mutable. 
- Construction is a no-copy operation, though the data is not modifiable. 
```cpp
    Tensor_View<int Dimensions, ValueType, Allocator>
    Scalar_View<ValueType, Allocator>
    Vector_View<ValueType, Allocator>
    Matrix_View<ValueType, Allocator>
    Cube_View<ValueType, Allocator>

```

#### Expression Types:
- A TensorXpr reference should be used in any function that will accept a Tensor or TensorExpression.
- Cannot be constructed directly, used only for method parameters. 
```cpp
    TensorXpr<int, T>
    ScalarXpr<T>
    VectorXpr<T>
    MatrixXpr<T>
    CubeXpr<T>
  ```
