# Streams and Contexts 



BlackCat_Tensors's support using Cuda streams. 
A context stores the required meta-data of handling computations of a Tensor.

In regards to Cuda, 
a context stores a unique cublasHandle_t, cudaStream_t, cudaEvent_t, and a memory_pool.

The cpu version contains a HostStream object and a memory_pool. 


A context by default uses the default stream. (The hostStream, behaves similar) 

Tensors using non-default streams have their operations run asynchrously to the host thread. 

The context used on expression is based upon the assignment operator of a single expression. 


```cpp

BC::Matrix<float> y, w, x;

y.create_stream();
w.create_stream();
x.create_stream();

y = w * x; //will use Y's Context and stream 
```

Contexts will be propagated to slices of a tensor. 

```cpp

y.get_context() == y[0].get_context(); //True 

```



```cpp

using Context = BC::context::implementation<system_tag>; 
Context context;

context.create_stream();			//calls cudaCreateStream
context.set_stream(cudaStream_t or HostStream)  //sets the stream 


```
