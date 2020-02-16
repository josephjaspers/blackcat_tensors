/*
 * Polymorphic_Allocator.h
 *
 *  Created on: Mar 1, 2019
 *      Author: joseph
 */

#ifndef BC_CORE_CONTEXT_MEMORY_MANAGER_POLYMORPHIC_ALLOCATOR_H_
#define BC_CORE_CONTEXT_MEMORY_MANAGER_POLYMORPHIC_ALLOCATOR_H_

#include <memory>
#include <typeinfo>
namespace bc {
namespace allocators {
/**
 * Similar to the C++17 std::pmr::polymorphic_allocator.
 *
 * The polymorphic_allocator accepts an Allocator object on construction
 * and uses polymorphism to create an expression_template 'allocator.' All calls of allocate/deallocate
 * are forwarded to the stated allocator. The polymorphic_allocator there-for, cannot be rebound.
 *
 */

namespace pa_detail {

template<class ValueType, class SystemTag>
struct Allocator_Base
{
	using system_tag = SystemTag;
	using value_type = ValueType;

	virtual
	value_type* allocate(std::size_t sz) = 0;
	virtual void deallocate(value_type* data, std::size_t sz) = 0;

	virtual Allocator_Base* clone() const = 0;
	virtual ~Allocator_Base() {};

	virtual bool operator == (const Allocator_Base& other) const = 0;
	virtual bool operator != (const Allocator_Base& other) const  {
		return !((*this) == other);
	}
	virtual std::string rebound_to_byte_hash() const = 0;
};

template<class Allocator>
struct Derived_Allocator:
	Allocator_Base<
			typename bc::allocator_traits<Allocator>::value_type,
			typename bc::allocator_traits<Allocator>::system_tag>
{
		using traits_type = bc::allocator_traits<Allocator>;
		using system_tag = typename traits_type::system_tag;
		using value_type = typename traits_type::value_type;

private:

	using parent_type = Allocator_Base<value_type, system_tag>;
	using self_type = Derived_Allocator<Allocator>;

	Allocator m_allocator;

public:

	template<class altT>
	struct rebind
	{
		using other = Derived_Allocator<
				typename traits_type::template rebind_alloc<altT>>;
	};

	Allocator_Base<value_type, system_tag>* clone() const override {
		return new Derived_Allocator(m_allocator);
	}

	Derived_Allocator() {}

	Derived_Allocator(const Allocator& alloc):
		m_allocator(alloc) {}

	Derived_Allocator(Allocator&& alloc):
		m_allocator(alloc) {}

	virtual ~Derived_Allocator() override {}


	static std::string static_hash()
	{
		return __PRETTY_FUNCTION__ ;
	}

	virtual std::string rebound_to_byte_hash() const final {
		using hash_t = std::conditional_t<
				std::is_same<value_type, bc::allocators::Byte>::value,
				self_type,
				typename rebind<bc::allocators::Byte>::other>;
		return hash_t::static_hash();
	}

	virtual value_type* allocate(std::size_t sz) override final {
		return m_allocator.allocate(sz);
	}

	virtual void deallocate(value_type* data, std::size_t sz) override final {
		m_allocator.deallocate(data, sz);
	}

	virtual bool operator == (const Allocator_Base<value_type, system_tag>& other) const override
	{
		//same derived class
		if (rebound_to_byte_hash() == other.rebound_to_byte_hash()) {
			if (traits_type::is_always_equal::value)
				return true;

			auto& cast_other = static_cast<const self_type&>(other);
			return m_allocator == cast_other.m_allocator;
		} else {
			return false;
		}
	}
};


}

template<class ValueType, class SystemTag>
struct Polymorphic_Allocator
{
	using system_tag = SystemTag;
	using value_type = ValueType;

private:

	template<class... Args>
	using Derived_Allocator = pa_detail::Derived_Allocator<Args...>;

	template<class... Args>
	using Allocator_Base = pa_detail::Allocator_Base<Args...>;

	using allocator_type =  Allocator_Base<value_type, system_tag>;
	using allocator_pointer_type = std::unique_ptr<allocator_type>;
	using default_allocator_type = bc::Allocator<value_type, system_tag>;

	allocator_pointer_type m_allocator;

public:

	template<class Allocator>
	Polymorphic_Allocator(const Allocator& alloc):
		m_allocator(allocator_pointer_type(
			new Derived_Allocator<Allocator>(alloc))) {}

	Polymorphic_Allocator():
		m_allocator(allocator_pointer_type(
			new Derived_Allocator<default_allocator_type>())) {}

	Polymorphic_Allocator(const Polymorphic_Allocator& pa):
		m_allocator(pa.m_allocator->clone()) {}

	Polymorphic_Allocator& operator = (const Polymorphic_Allocator& other) {
		this->set_allocator(other.m_allocator->clone());
		return *this;
	}

	Polymorphic_Allocator& operator = (Polymorphic_Allocator&& other) {
		this->m_allocator = std::move(other.m_allocator);
		return *this;
	}

	value_type* allocate(std::size_t sz) {
		return m_allocator->allocate(sz);
	}

	void deallocate(value_type* data, std::size_t sz) {
		m_allocator->deallocate(data, sz);
	}

	template<class Allocator>
	void set_allocator(const Allocator& alloc)
	{
		using traits = bc::allocator_traits<Allocator>;
		using alloc_rb_t = typename traits::template rebind_alloc<value_type>;

		auto alloc_rebound = alloc_rb_t(alloc);

		m_allocator = allocator_pointer_type(
			new Derived_Allocator<alloc_rb_t>(alloc_rebound));
	}

	template<class AltT>
	bool operator == (const Polymorphic_Allocator<AltT, system_tag>& other)
	{
		return *m_allocator == *(other.m_allocator);
	}

	template<class AltT>
	bool operator != (const Polymorphic_Allocator<AltT, system_tag>& other)
	{
		return !(*this == other);
	}


};

}
}



#endif /* POLYMORPHIC_ALLOCATOR_H_ */
