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
namespace allocator {
namespace fancy {

template<class ValueType, class SystemTag>
struct Polymorphic_Allocator {

	using system_tag = SystemTag;
	using value_type = ValueType;
	using default_allocator_t = BC::Allocator<system_tag, value_type>;

private:

	struct Virtual_Allocator {
		virtual value_type* allocate(std::size_t sz) = 0;
		virtual void deallocate(value_type* memptr, std::size_t sz) = 0;
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


		Virtual_Allocator* clone() const {
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

		virtual void deallocate(value_type* memptr, std::size_t sz) override final {
			retrieve_allocator(m_allocator).deallocate(memptr, sz);
		}
	};

	std::unique_ptr<Virtual_Allocator> m_allocator;

public:


	template<class Allocator>
	Polymorphic_Allocator(const Allocator& alloc)
	: m_allocator(std::unique_ptr<Virtual_Allocator>(new Derived_Allocator<Allocator>(alloc))) {}

	Polymorphic_Allocator()
	: m_allocator(std::unique_ptr<Virtual_Allocator>(new Derived_Allocator<default_allocator_t>())) {}

	Polymorphic_Allocator(const Polymorphic_Allocator& pa)
	: m_allocator(pa.m_allocator.get()->clone()) {}

	value_type* allocate(std::size_t sz) {
		return m_allocator.get()->allocate(sz);
	}
	void deallocate(value_type* memptr, std::size_t sz) {
		m_allocator.get()->deallocate(memptr, sz);
	}

	template<class Allocator>
	void set_allocator(const Allocator& alloc) {
		using allocator_t = typename Allocator::template rebind<BC::allocator::Byte>::other;
		m_allocator = std::unique_ptr<Virtual_Allocator>(
				new Derived_Allocator<allocator_t>(alloc));
	}

	auto& get_allocator() {
		return m_allocator.get().retrieve_allocator();
	}

	auto& get_allocator() const {
		return m_allocator.get().retrieve_allocator();
	}

};

}
}
}



#endif /* POLYMORPHIC_ALLOCATOR_H_ */
