# Algorithms

BlackCat_Tensor's offer a wide range of the `std` library's algorithms. These methods are called through the `BC` namespace. IE BC::for_each. 

BlackCat_Tensor's does not implement any of these algorithms itself, instead forwarding to either the std implementation or thrust's implementation depending on how memory is allocated.

```
BC::Matrix<float, BC::Cuda> dev_mat(3,3);

//BC::for_each will forward to thrust's (Cuda) implementation.
BC::for_each(dev_mat.begin(), dev_mat.end(), your_function); 


BC::Matrix<float, BC::Basic_Allocator> host_mat(3,3);

//BC::for_each will forward to the standard lib's implementation.
BC::for_each(dev_mat.begin(), dev_mat.end(), your_function); 
```

Using BC::algortihms is preferable to directly using to std or thrust's implementation as it enables user's to write allocation-generic code. 

Example:
	```
	template<class alloc>
	void logistic_function(BC::Matrix<float, alloc>& mat) {

		struct logistic_functor {
			__host__ __device__
			void operator() (scalar_t& scalar) {
				scalar =  1 / (1 + exp(-scalar));
			}		
	
		} logistc; 

		BC::for_each(mat.begin(), mat.end(), logistic); 
	}

	```

	Here we created a method that applies the logistic function to each element of a matrix. This function can accept Tensors allocated on either the GPU or CPU. If we used std::for_each, or thrust::for_each, we would have to overload for_each (pun) allocator we may want to use. 


The supported algorithms:

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





