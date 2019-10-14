/*
 * TypeMap.h
 *
 *  Created on: Oct 13, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSORS_UTILITY_TYPEMAP_H_
#define BLACKCAT_TENSORS_UTILITY_TYPEMAP_H_

#include <unordered_map>
#include <memory>

namespace BC {
namespace utility {
namespace experimental {

class Type_Map {

	using any_ptr = std::shared_ptr<void>;
	std::unordered_map<std::string, any_ptr> m_any_map;

	template<class T>
	static std::string hash(BC::traits::Type<T> type_key) {
		return typeid(T).name();
	}

public:

	template<class T>
	bool contains(BC::traits::Type<T> key) const {
		return m_any_map.find(hash(key)) != m_any_map.end();
	}

	template<class T>
	auto& operator [] (BC::traits::Type<T> key) {
		static auto deleter = [](void* data) {
			delete reinterpret_cast<T*>(data);
		};

		if (!contains(key)) {
			void* data = reinterpret_cast<void*>(new T());
			m_any_map[hash(key)] = any_ptr(data, deleter);
		}

		return *reinterpret_cast<T*>(m_any_map[hash(key)].get());
	}
};

}
}
}



#endif /* TYPEMAP_H_ */
