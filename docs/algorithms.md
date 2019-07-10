# Algorithms

BlackCat_Tensor's offer a wide range of the standard library's algorithms. 
These methods are called through the `BC` namespace.

BlackCat_Tensor's does not implement any of these algorithms itself, instead forwarding to either the std implementation or thrust's implementation depending on how memory is allocated.

The first argument must always be a stream argument, the possible arguments are: BC::Stream<host_tag>, BC::Stream<device_tag>, cudaStream_t.


```cpp
BC::Matrix<float, BC::Cuda_Allocator<float>> dev_mat(3,3);
BC::algorithms::for_each(dev_mat.get_stream(), dev_mat.begin(), dev_mat.end(), your_function);  //will call thrust::for_each

BC::Matrix<float, BC::Basic_Allocator<float>> host_mat(3,3);
BC::algorithms::for_each(dev_mat.get_stream(), dev_mat.begin(), dev_mat.end(), your_function); //will call std::for_each
```

Convenience definitions are given for each of the supported algorithms.

IE 
```cpp
BC::Matrix<float> dev_mat(3,3);
BC::algorithms::for_each(dev_mat, your_function);  //will call thrust::for_each

```



Using BC::algortihms is preferable to directly using to std or thrust's implementation as it enables user's to write allocation-generic code. Here we created a method that applies the sigmoid function to each element of a matrix. 

```cpp
struct Sigmoid {
	template<class ValueType> __host__ __device__
	void operator() (ValueType& scalar) {
		scalar =  1 / (1 + exp(-scalar));
	}		

}; 

template<class ValueType, class Allocator>
void logistic_function(BC::Matrix<ValueType, Allocator>& matrix) {
	BC::for_each(matrix, Sigmoid()); 
}
```

This function can accept Tensors allocated on either the GPU or CPU. If we used std::for_each, or thrust::for_each, we would have to write two seperate methods for the host and device. 


#####  The supported algorithms:
    
    accumulate
    count
    count_if
    find
    find_if
    find_if_not
    for_each
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
    sort
    stable_sort
    max_element
    min_element
    max
    min
    minmax_element
    equal





