/*

 * Common.h
 *
 *  Created on: Jan 13, 2019
 *      Author: joseph
 */

#ifndef BC_STREAMS_COMMON_H_
#define BC_STREAMS_COMMON_H_

#include <memory>
#include <iostream>
#include "Polymorphic_Allocator.h"
#include "HostQueue.h"
#include "Workspace.h"

namespace BC {
namespace context {


class Host {

	struct Contents {
		HostQueue m_stream;
		Workspace<host_tag> m_workspace;
	};

	static std::shared_ptr<Contents> get_default_contents() {
		thread_local std::shared_ptr<Contents> default_contents =
					std::shared_ptr<Contents>(new Contents());

		return default_contents;
	}
	std::shared_ptr<Contents> m_contents = get_default_contents();

public:

	using system_tag = host_tag;

	Host() =default;
	Host(const Host&)=default;
	Host(Host&&)=default;

    template<class scalar_t, int value>
    static scalar_t scalar_constant() {
    	return value;
    }

    Workspace<host_tag>& get_allocator() {
    	return m_contents.get()->m_workspace;
    }

	 template<class scalar_t>
	 scalar_t scalar_alpha() {
		 return scalar_t();
	 }

	 Host& get_stream() {
		 return *this;
	 }
	 const Host& get_stream() const {
		 return *this;
	 }

	bool is_default_stream() {
		return m_contents == get_default_contents();
	}

	void create_stream() {
		m_contents = std::shared_ptr<Contents>(new Contents());
		m_contents.get()->m_stream.init();
	}

	void destroy_stream() {
		m_contents = get_default_contents();
	}

	void sync_stream() {
		//** Pushing a job while syncing is undefined behavior.
		if (!is_default_stream())
			m_contents.get()->m_stream.synchronize();
	}

	void set_stream(Host& stream_) {
		this->m_contents = stream_.m_contents;
	}

	template<class function_lambda>
	void push_job(function_lambda functor) {
		if (this->is_default_stream()) {
			functor();
		} else {
			m_contents.get()->m_stream.push(functor);
		}
	}

    bool operator == (const Host& dev) {
    	return m_contents == dev.m_contents;
    }
    bool operator != (const Host& dev) {
    	return m_contents != dev.m_contents;
    }
};


}
}


#endif /* COMMON_H_ */
