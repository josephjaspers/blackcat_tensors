/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_MEMORY_H_
#define BLACKCAT_MEMORY_H_

#include "BlackCat_Allocator.h"
#include <memory>
#include <mutex>

namespace BC {
namespace memory {
namespace detail {

struct Default_Deleter {
	template<class T>
	void operator ()(T* ptr) {
		delete ptr;
	}
} default_deleter;


}


template<class ValueType>
class atomic_shared_ptr {

	mutable std::mutex locker;
	std::shared_ptr<ValueType> m_ptr;


public:

	template<class... Args>
	explicit atomic_shared_ptr(Args&&... args) : m_ptr(args...) {};

	atomic_shared_ptr(atomic_shared_ptr&& ptr) {
		*this = ptr;
	}
	
	atomic_shared_ptr(const atomic_shared_ptr& ptr) {
		*this = ptr;
	}
	
	atomic_shared_ptr() = default;
	
	template<class... Args>
	atomic_shared_ptr(const Args&... args) : m_ptr(args...) {};

	struct wrapper {
		std::mutex& locker;
		std::shared_ptr<ValueType> m_ptr;
		~wrapper() {
			locker.unlock();
		}

		auto operator * () const  { return m_ptr.get(); }
		auto operator * () { return m_ptr.get(); }

		auto operator ->() const { return m_ptr.get(); }
		auto operator ->()  { return m_ptr.get(); }
	};


	template<class Y, class Deleter=detail::Default_Deleter, class Allocator=std::allocator<ValueType>>
	void reset(Y* ptr, Deleter del=detail::default_deleter, Allocator alloc=Allocator()) {
		locker.lock();
		m_ptr.reset(ptr, del, alloc);
		locker.unlock();
	}

	wrapper get() {
		locker.lock();
		return wrapper {locker, m_ptr};
	}
	const wrapper get() const {
		locker.lock();
		return wrapper {locker, m_ptr};
	}

	auto operator ->() { return this->get(); }
	auto operator ->() const { return this->get(); }

	bool operator == (const atomic_shared_ptr<ValueType>& ptr) {
		return ptr.m_ptr == this->m_ptr;
	}
	bool operator != (const atomic_shared_ptr<ValueType>& ptr) {
			return !(*this == ptr);
		}

	atomic_shared_ptr& operator = (const atomic_shared_ptr<ValueType>& ptr) {
		std::lock_guard<std::mutex> lck1(this->locker);
		std::lock_guard<std::mutex> lck2(ptr.locker);
		this->m_ptr = ptr.m_ptr;
		return *this;
	}
};

}
}
#endif /* BLACKCAT_MEMORY_H_ */
