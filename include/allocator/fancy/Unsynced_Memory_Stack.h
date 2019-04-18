#ifndef BC_ALLOCATORS_FANCY_UNSYNCED_MEMORY_STACK_H_
#define BC_ALLOCATORS_FANCY_UNSYNCED_MEMORY_STACK_H_

namespace BC {
namespace allocators {

template<class value_type_, class derived>
class Unsynced_Memory_Stack {
public:

	using value_type = value_type_;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

private:

	value_type* pool = nullptr;
	std::size_t index = 0;
	std::size_t max_size;

protected:

	value_type* get_pool() { return pool; }
	const value_type* get_pool() const { return pool; }
	value_type*& get_pool_ref() { return pool; }
	const value_type*& get_pool_ref() const { return pool; }


public:

	value_type* allocate(std::size_t sz) {

		if (index + sz > max_size) {
			throw std::bad_alloc("Exceeded maximum allocation");
		}

		value_type* memptr = &pool[index];
		index += sz;

		return memptr;
	}

	void deallocate(value_type* memptr, std::size_t sz) {
		pool -= sz;
		index -= sz;

#ifndef NDEBUG
		if (pool != memptr) {
			throw std::invalid_argument("attempting to deallocate memory out of order");
		}
#endif
	}

    constexpr bool operator == (const Unsynced_Memory_Stack& lv) {
    	return this->pool == lv.pool;
    }
    constexpr bool operator != (const Unsynced_Memory_Stack& lv) {
    	return this->pool != lv.pool;
    }


protected:
	Unsynced_Memory_Stack(std::size_t max_size_) ; max_size(max_size_) {};
	Unsynced_Memory_Stack(const Unsynced_Memory_Stack&) = default;
	Unsynced_Memory_Stack(Unsynced_Memory_Stack&&) = default;
};


template<class scalar_t>
struct Host_Unsynced_Memory_Stack : Unsynced_Memory_Stack<scalar_t,
	Host_Unsynced_Memory_Stack<scalar_t>> {

	using parent = Unsynced_Memory_Stack<scalar_t,
			Host_Unsynced_Memory_Stack<scalar_t>>;
	using system_tag = host_tag;
    using is_always_equal = std::false_type;

    template<class alt_scalar_t>
    struct rebind { using other = Host_Unsynced_Memory_Stack<alt_scalar_t>; };


    Host_Unsynced_Memory_Stack(std::size_t sz)
    : parent(sz) {
    	this->get_pool_ref() = new scalar_t[sz];
    }

    Host_Unsynced_Memory_Stack(Host_Unsynced_Memory_Stack&&) = default;
    Host_Unsynced_Memory_Stack(const Host_Unsynced_Memory_Stack&) = default;
};

#ifdef __CUDACC__
template<class scalar_t>
struct Device_Unsynced_Memory_Stack : Unsynced_Memory_Stack<scalar_t,
Device_Unsynced_Memory_Stack<scalar_t>> {

	using parent = Unsynced_Memory_Stack<scalar_t,
			Device_Unsynced_Memory_Stack<scalar_t>>;
	using system_tag = device_tag;
    using is_always_equal = std::false_type;

    template<class alt_scalar_t>
    struct rebind { using other = Host_Unsynced_Memory_Stack<alt_scalar_t>; };


    Device_Unsynced_Memory_Stack(std::size_t sz)
    : parent(sz) {
        cudaMalloc((void**) this->get_pool_ref(), sizeof(scalar_t) * sz);
    }

    Device_Unsynced_Memory_Stack(Device_Unsynced_Memory_Stack&&) = default;
    Device_Unsynced_Memory_Stack(const Device_Unsynced_Memory_Stack&) = default;
};

template<class scalar_t>
struct Cuda_Managed_Unsynced_Memory_Stack : Unsynced_Memory_Stack<scalar_t,
Cuda_Managed_Unsynced_Memory_Stack<scalar_t>> {

	using parent = Unsynced_Memory_Stack<scalar_t,
			Cuda_Managed_Unsynced_Memory_Stack<scalar_t>>;
	using system_tag = device_tag;
    using is_always_equal = std::false_type;

    template<class alt_scalar_t>
    struct rebind { using other = Host_Unsynced_Memory_Stack<alt_scalar_t>; };


    Cuda_Managed_Unsynced_Memory_Stack(std::size_t sz)
    : parent(sz) {
        cudaMallocManaged((void**) this->get_pool_ref(), sizeof(scalar_t) * sz);
    }

    Cuda_Managed_Unsynced_Memory_Stack(Cuda_Managed_Unsynced_Memory_Stack&&) = default;
    Cuda_Managed_Unsynced_Memory_Stack(const Cuda_Managed_Unsynced_Memory_Stack&) = default;
};

#endif

}
}



#endif
