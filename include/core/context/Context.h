/*
 * Context.h
 *
 *  Created on: Jan 24, 2019
 *      Author: joseph
 */

#ifndef BC_CONTEXT_CONTEXT_H_
#define BC_CONTEXT_CONTEXT_H_

#include "Host.h"
#include "Device.h"

BC_DEFAULT_MODULE_BODY(context)

namespace BC {

template<class system_tag>  //push into BC namespace
using Context = context::template implementation<system_tag>;


template<class Allocator, class Context>
struct Full_Context : Context {

	using context_t = Context;
	using allocator_t = Allocator;

	Allocator& m_allocator;

	Full_Context(const Full_Context&) = default;
	Full_Context(Full_Context&&) = default;
	Full_Context(Allocator& alloc_, const Context& context)
	: Context(context), m_allocator(alloc_) {}

	const Allocator& get_allocator() const { return m_allocator; }
		  Allocator& get_allocator() 	   { return m_allocator; }

};
namespace context {

template<class Allocator, class Context>
auto make_full_context(Allocator& alloc, Context& cont) {
	return Full_Context<Allocator, Context>(alloc, cont);
}

template<class Allocator, class Context>
using full_context_t = Full_Context<Allocator, Context>;

} //end of context

}



#endif /* CONTEXT_H_ */
