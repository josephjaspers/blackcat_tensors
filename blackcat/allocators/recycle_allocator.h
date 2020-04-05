/*
 * RecyclerAllocator.h
 *
 *  Created on: Sep 21, 2019
 *	  Author: joseph
 */

#ifndef BLACKCATTENSORS_ALLOCATORS_FANCY_H_
#define BLACKCATTENSORS_ALLOCATORS_FANCY_H_

#include "basic_allocators.h"
#include <unordered_map>
#include <vector>
#include <mutex>

namespace bc {
namespace allocators {

//TODO make friend class and private members
struct Recycle_Allocator_Globals
{
	template<class SystemTag>
	static auto& get_recycler(SystemTag=SystemTag()) {
		static std::unordered_map<bc::size_t, std::vector<Byte*>> m_recycler;
		return m_recycler;
	}

	template<class SystemTag>
	static auto& get_locker(SystemTag=SystemTag()) {
		static std::mutex m_locker;
		return m_locker;
	}

	template<class SystemTag>
	static void clear_recycler(SystemTag system=SystemTag()) {
		std::lock_guard<std::mutex> locker(get_locker(system));
		bc::Allocator<Byte, SystemTag> allocator;

		auto& recycler = get_recycler(system);
		for (const auto& kv : recycler) {
			std::size_t ptr_sz = kv.first;
			const std::vector<Byte*>& ptrs = kv.second;

			for (auto ptr : ptrs)
				allocator.deallocate(ptr, ptr_sz);
		}
		recycler.clear();
	}

};


template<
		class T,
		class SystemTag,
		class AlternateAllocator=Allocator<Byte, SystemTag>>
struct Recycle_Allocator {

	using system_tag = SystemTag;	//BC tag
	using value_type = T;
	using pointer = value_type*;
	using const_pointer = const value_type*;
	using size_type = bc::size_t;
	using propagate_on_container_copy_assignment = std::false_type;
	using propagate_on_container_move_assignment = std::false_type;
	using propagate_on_container_swap = std::false_type;
	using is_always_equal = std::true_type;

	static_assert(std::is_same<Byte, typename AlternateAllocator::value_type>::value,
			"AlternateAllocator of Recycle_Allocator must have Byte value_type");

	static_assert(std::is_same<SystemTag, typename AlternateAllocator::system_tag>::value,
			"AlternateAllocator of Recycle_Allocator must have same system_tag");

private:

	AlternateAllocator m_allocator;

	static auto& get_recycler() {
		return Recycle_Allocator_Globals::get_recycler(system_tag());
	}
	static auto& get_locker() {
		return Recycle_Allocator_Globals::get_locker(system_tag());
	}

public:

	template<class altT>
	struct rebind {
		using other = Recycle_Allocator<altT, SystemTag, AlternateAllocator>;
	};

	Recycle_Allocator()=default;
	Recycle_Allocator(const Recycle_Allocator&)=default;
	Recycle_Allocator(Recycle_Allocator&&)=default;
	Recycle_Allocator& operator=(const Recycle_Allocator&)=default;
	Recycle_Allocator& operator=(Recycle_Allocator&&)=default;


	template<class U>
	Recycle_Allocator(const Recycle_Allocator<U, SystemTag, AlternateAllocator>& other) {}


	T* allocate(bc::size_t size) {
		if (size == 0) { return nullptr; }

		std::lock_guard<std::mutex> lck(get_locker());
		size *= sizeof(value_type);

		auto& recycler = get_recycler();

		if (recycler.find(size) != recycler.end() && !recycler[size].empty()) {
			T* data = reinterpret_cast<value_type*>(recycler[size].back());
			recycler[size].pop_back();
			return data;
		} else {
			return reinterpret_cast<value_type*>(m_allocator.allocate(size));
		}
	}

	void deallocate(T* ptr, bc::size_t size) {
		if (size == 0 || ptr==nullptr) { return; }
		std::lock_guard<std::mutex> lck(get_locker());
		size *= sizeof(value_type);
		get_recycler()[size].push_back(reinterpret_cast<Byte*>(ptr));
	}

	void clear_cache() {
		std::lock_guard<std::mutex> lck(get_locker());
		for (auto kv : get_recycler()) {
			for (Byte* ptr: kv.second) {
				m_allocator.deallocate(ptr, kv.first);
			}
		}
	}

	template<class U>
	constexpr bool operator == (
		const Recycle_Allocator<U, SystemTag, AlternateAllocator>&) const {
		return true;
	}
	template<class U>
	constexpr bool operator != (
		const Recycle_Allocator<U, SystemTag, AlternateAllocator>&) const {
		return false;
	}
};

}
}




#endif /* RECYCLERALLOCATOR_H_ */
