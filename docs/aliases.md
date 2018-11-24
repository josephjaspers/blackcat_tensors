# Aliases 

BlackCat_Tensor's defines a numerous amount of user-types.

  ```cpp
  scalar_t    --> the scalar_type of the Tensor
  allocator_t --> the allocation methods of the Tensor

  //Standard Tensor Types
    Tensor<int x, scalar_t, allocator_t> 
    Scalar<scalar_t, allocator_t>
    Vector<scalar_t, allocator_t>
    Matrix<scalar_t, allocator_t>

  //View Types - accepts an Array to construct. Shares the internal_pointer and is not mutable. 
  //             Comparable std::string_view, in which construction is a no-copy, though the data is not modifiable.
    Tensor_View<int x, scalar_t, allocator_t>
    Scalar_View<scalar_t, allocator_t>
    Vector_View<scalar_t, allocator_t>
    Matrix_View<scalar_t, allocator_t>

  //Shared Types - accepts a non-const Array to construct. Shares the internal_pointer but is mutable
  //               Identical to Tensor_View, except mandates a non-const Tensor on construction and allos modification of the internal data.
    Tensor_Shared<int x, scalar_t, allocator_t> 
    Scalar_Shared<scalar_t, allocator_t> 
    Vector_Shared<scalar_t, allocator_t>
    Matrix_Shared<scalar_t, allocator_t>

 //expression Types - any tensor that is expression or array of the apropriate dimensionality
 //                   Used for parameter arguments, cannot be constructed directly.
    expr::Tensor<int> 
    expr::Scalar
    expr::Vector
    expr::Matrix
  ```
