# Streams

BlackCat_Tensors's support using Cuda streams. 
A stream stores the required meta-data of handling computations of a Tensor.

In regards to Cuda, 
a stream stores a unique cublasHandle_t, cudaStream_t, cudaEvent_t, and a memory_pool.
The cpu version contains a HostStream object and a memory_pool. 

A stream by default uses the default stream. (The hostStream, behaves similar) 
Tensors using non-default streams have their operations run asynchrously to the host thread. 
The stream used for any given expression is based upon the assignment operator of a single expression. 


```cpp

bc::Matrix<float> y, w, x;

y.get_stream().create();
w.get_stream().create();
x.get_stream().create();

y = w * x; //will use y's stream 
```

Streams will be propagated to slices of a tensor. 

```cpp
y.get_stream() == y[0].get_stream();
```

```cpp

bc::Stream<system_tag> stream;

stream.create();		  //calls cudaCreateStream
stream.set_stream(cudaStream_t);  //sets the stream 
stream.set_stream(another_bc_stream_object); 

```


#### Methods 

| return type | method name | parameters | documentation |
| --- | --- | --- | --- |
| bool | is_default | --- | returns if the stream is the default stream |
| void | create | --- | creates a stream inplace, the stream will refer to a new non-default stream |
| void | destroy | --- | destroys the stream, the stream used will be the default stream |
| void | sync | --- | synchronizes the host stream with the current stream | 
| void | set_stream | cudaStream_t or bc::Stream<system_tag> | Sets the stream to the same stream as the parameter | 
| void | record_event | --- | Records an event (using a bc::HostEvent object or cudaEvent_t object) | 
| void | wait_event | bc::Stream | Causes the current stream to wait on the last recorded event of the parameter-stream | 
| void | wait_stream | bc::Stream | Shorthand for recording an event on the parameter-stream and synchronizing on that stream |
| void | enqueue_callback | Functor |Enques a host-call back function into the stream, a device stream will not wait for the completion of the host event to continue running | 
| Workspace<ValueType> | get_allocator_rebound<value_type> | --- | Returns a bc::fancy::Workspace allocator contained inside the stream. | 
 

