/*
 * Polymorphic_Allocator.h
 *
 *  Created on: Mar 1, 2019
 *      Author: joseph
 */

#ifndef BC_CORE_CONTEXT_MEMORY_MANAGER_POLYMORPHIC_ALLOCATOR_H_
#define BC_CORE_CONTEXT_MEMORY_MANAGER_POLYMORPHIC_ALLOCATOR_H_

#include <memory>

namespace BC {
namespace allocators {
/**
 * Similar to the C++17 std::pmr::polymorphic_allocator.
 *
 * The polymorphic_allocator accepts an Allocator object on construction
 * and uses polymorphism to create an internal 'allocator.' All calls of allocate/deallocate
 * are forwarded to the stated allocator. The polymorphic_allocator there-for, cannot be rebound.
 *
 */
template<class SystemTag, class ValueType>
struct Polymorphic_Allocator {

	using system_tag = SystemTag;
	using value_type = ValueType;
	using default_allocator_t = BC::Allocator<system_tag, value_type>;

private:

	struct Virtual_Allocator {
		virtual value_type* allocate(std::size_t sz) = 0;
		virtual void deallocate(value_type* data, std::size_t sz) = 0;
		virtual Virtual_Allocator* clone() const = 0;
	};

	template<class Allocator>
	class Derived_Allocator :
			public Virtual_Allocator {


		Allocator m_allocator;

		static_assert(std::is_same<value_type, typename Allocator::value_type>::value,
				"SUPPLIED ALLOCATOR VALUE_TYPE MUST BE SAME AS POLYMORPHIC ALLOCATOR VALUE_TYPE");

		static_assert(std::is_same<system_tag, typename BC::allocator_traits<Allocator>::system_tag>::value,
				"SUPPLIED ALLOCATOR VALUE_TYPE MUST BE SAME AS POLYMORPHIC ALLOCATOR VALUE_TYPE");

		public:


		Virtual_Allocator* clone() const override {
			return new Derived_Allocator(m_allocator);
		}

		template<class Alloc_t>
		auto& retrieve_allocator(std::shared_ptr<Alloc_t>& this_allocator) {
			BC_ASSERT(this_allocator.get(),
					"Attempting to retrieve allocator from shared_ptr resulted in nullptr");
			return *(this_allocator.get());
		}
		template<class Alloc_t>
		auto& retrieve_allocator(Alloc_t& this_allocator) {
			return this_allocator;
		}
		template<class Alloc_t>
		auto& retrieve_allocator(Alloc_t* this_allocator) {
			return *this_allocator;
		}
		Derived_Allocator() = default;
		Derived_Allocator(const Allocator& alloc)
		: m_allocator(alloc) {}
		Derived_Allocator(Allocator&& alloc)
		: m_allocator(alloc) {}


		virtual value_type* allocate(std::size_t sz) override final {
			return retrieve_allocator(m_allocator).allocate(sz);
		}

		virtual void deallocate(value_type* data, std::size_t sz) override final {
			retrieve_allocator(m_allocator).deallocate(data, sz);
		}
	};

	std::unique_ptr<Virtual_Allocator> m_allocator;

public:


	template<class Allocator>
	Polymorphic_Allocator(const Allocator& alloc):
		m_allocator(std::unique_ptr<Virtual_Allocator>(new Derived_Allocator<Allocator>(alloc))) {}

	Polymorphic_Allocator():
		m_allocator(std::unique_ptr<Virtual_Allocator>(new Derived_Allocator<default_allocator_t>())) {}

	Polymorphic_Allocator(const Polymorphic_Allocator& pa):
		m_allocator(pa.m_allocator.get()->clone()) {}

	Polymorphic_Allocator& operator = (const Polymorphic_Allocator& alloc) {
		this->set_allocator(alloc.m_allocator->clone());
		return *this;
	}

	Polymorphic_Allocator& operator = (Polymorphic_Allocator&& alloc) {
		this->m_allocator = std::move(alloc.m_allocator);
		return *this;
	}

	value_type* allocate(std::size_t sz) {
		return m_allocator.get()->allocate(sz);
	}
	void deallocate(value_type* data, std::size_t sz) {
		m_allocator.get()->deallocate(data, sz);
	}

	template<class Allocator>
	void set_allocator(const Allocator& alloc) {
		///Change the underlying allocator
		using allocator_t = typename Allocator::template rebind<BC::allocators::Byte>::other;
		m_allocator = std::unique_ptr<Virtual_Allocator>(
				new Derived_Allocator<allocator_t>(alloc));
	}
};

}
}



#endif /* POLYMORPHIC_ALLOCATOR_H_ */
