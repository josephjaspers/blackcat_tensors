# Iterators 

BlackCat_Tensors defines two types of iterators, coefficient-wise and n-dimensional. 

----------------------------------------------------------------------------------------------

##### Coefficient-Wise 
The coefficient-wise iterator returns references of the `value_type` of the tensor.

```cpp
int main() {

    BC::Matrix<float> matrix(5, 5); 
    
    for (auto it = matrix.cw_begin(); it != matrix.cw_end(); ++it) {
      //do work 
    }
    
    //identical
    for (float& val : matrix.cw_iter()) {
      //do work 
    }
}
```
**Warning** Tensor's allocated via non-managed Cuda memory may not normally use the coefficient-wise iterator. (As it causes dereferencing a device pointer from the host). User's may opt to use `BC` [algorithms](https://github.com/josephjaspers/BlackCat_Tensors/blob/master/docs/algorithms.md) to meet their needs.   

----------------------------------------------------------------------------------------------
  
    
   
##### N-Dimensional     
The n-dimensional iterator returns slices of the current tensor.

```cpp

int main() { 

  BC::Cube<float> cube(3,3,3); 

  for (auto mat_iter = cube.begin(); mat_iter != cube.end(); ++mat_iter) {       
    for (auto vec_iter = (*mat_iter).begin(); vec_iter != (*mat_iter).end(); ++vec_iter) {        
      for (auto scalar_iter = (*vec_iter).begin(); scalar_iter != (*vec_iter).end(); ++scalar_iter) {
         //do work 
      }
    }
  }

//identical
  for (auto matrix : cube.nd_iter()) {       
    for (auto vec : cube.nd_iter()) {        
      for (float& scalar : vec) {
         //do work 
      }
    }
  }
}
```

**Note** Calling `nd_iter()` (regular `begin`/`end`) on a Vector returns Scalar 'view' objects, while calling `cw_iter` returns references to the underlying data type. The `cw_iter` is used for calling `std` style algorithms while the `nd_iter` is used for most general use cases.

----------------------------------------------------------------------------------------------
##### Std-Style Iterators

Formal `std` style iterators are supported; using `begin` and `end`. 

```cpp
BC::Matrix<float> mat; 

  mat.begin();        
  mat.end();
  mat.rbegin();
  mat.rend();

  mat.nd_begin();
  mat.nd_end();
  mat.nd_rbegin();
  mat.nd_rend();
```
#### Utility Iterators
Convienant iterator proxies which support start and end ranges.

```cpp
BC::Matrix<float> mat; 

  for (float& scalar : mat.cw_iter(start, finish)) {
    //do work
  }
  
  for (auto vec : mat.nd_iter(start, finish)) {
    //do work
  }

  for (auto vec : mat.reverse_nd_iter(finish, start)) {
    //do work
  }

  //reverse iterators are also supported.
  for (auto& float : mat.reverse_cw_iter(finish, start)) {
    //do work
  }
  


```
