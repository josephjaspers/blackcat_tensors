# Streams and Streams 



BlackCat_Tensors's support using Cuda streams. 
A stream stores the required meta-data of handling computations of a Tensor.

In regards to Cuda, 
a stream stores a unique cublasHandle_t, cudaStream_t, cudaEvent_t, and a memory_pool.

The cpu version contains a HostStream object and a memory_pool. 


A stream by default uses the default stream. (The hostStream, behaves similar) 

Tensors using non-default streams have their operations run asynchrously to the host thread. 

The stream used on expression is based upon the assignment operator of a single expression. 


```cpp

BC::Matrix<float> y, w, x;

y.create_stream();
w.create_stream();
x.create_stream();

y = w * x; //will use Y's Stream and stream 
```

Streams will be propagated to slices of a tensor. 

```cpp

y.get_stream() == y[0].get_stream(); //True 

```



```cpp

using Stream = BC::streams::implementation<system_tag>; 
Stream stream;

stream.create_stream();			//calls cudaCreateStream
stream.set_stream(cudaStream_t or HostStream)  //sets the stream 


```
