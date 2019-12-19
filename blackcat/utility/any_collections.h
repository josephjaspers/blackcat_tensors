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
#include <typeinfo>

namespace BC {
namespace utility {

template<class T>
using Type = BC::traits::Type<T>;

class Any_Set {

	using any_ptr = std::shared_ptr<void>;
	std::unordered_map<std::string, any_ptr> m_any_map;

	template<class T>
	static std::string hash(Type<T> type_key) {
		return typeid(T).name();
	}

public:

	template<class T>
	bool contains(Type<T> key) const {
		return m_any_map.find(hash(key)) != m_any_map.end();
	}

	template<class T>
	auto& operator [] (Type<T> key) {
		static auto deleter = [](void* data) {
			delete reinterpret_cast<T*>(data);
		};

		if (!contains(key)) {
			void* data = reinterpret_cast<void*>(new T());
			m_any_map[hash(key)] = any_ptr(data, deleter);
		}

		return *reinterpret_cast<T*>(m_any_map[hash(key)].get());
	}

	template<class T>
	auto& at (Type<T> key) {
		return this->operator[](key);
	}


	template<class T, class... DefaultArgs>
	auto& get(Type<T> key, DefaultArgs&&... args) {
		if (!contains(key)) {
			return m_any_map[hash(key)] = T(std::forward(args)...);
		} else {
			return m_any_map[hash(key)];
		}
	}

	template<class T>
	auto& get(Type<T> key, T&& value) {
		if (!contains(key)) {
			return m_any_map[hash(key)] = value;
		} else {
			return m_any_map[hash(key)];
		}
	}

	int empty() const { return m_any_map.empty(); }
	int size() const { return m_any_map.size(); }
	int max_size() const { return m_any_map.size(); }

	auto begin() const { return m_any_map.begin(); }
	auto end() const { return m_any_map.end(); }

	auto begin() { return m_any_map.begin(); }
	auto end() { return m_any_map.end(); }
	auto cbegin() { return m_any_map.cbegin(); }
	auto cend() { return m_any_map.cend(); }
};



template<char... Chars>
struct Name {
	std::string name() {
		return "";
	}
};

template<char C, char... Chars>
struct Name<C, Chars...> {
	static std::string name() {
		return "C" + Name<Chars...>::name();
	}
};

template<class Key, class Value>
struct Any_Key {
	using key_type = Key;
	using value_type = Value;
};

/**
 * Any_Map stores a buck of std::shared_ptr<void>.
 * Elements are retrieved through:
 *
 * myMap[Any_Key<K, V>()]
 *
 * It is recommended to use "Name" with Any_Key to emulate constexpr-strings
 *
 * myMap[Any_Key<Name<'K','E','Y'>, ValueType>]
 *
 * Once CUDA supporst C++17 (which supports constexpr strings as template args)
 * we will switch Any_Key to simply being a <String, ValueType>.
 *
 * The operator[] is a template argument, which enables casting to the correct
 * type without any dynamic checks. This results in efficient access to 'any'
 * type within a pseudo-heterogeneous container.
 *
 *
 */
class Any_Map {

	using any_ptr = std::shared_ptr<void>;
	std::unordered_map<std::size_t, any_ptr> m_any_map;

	template<class K, class V>
	static std::size_t hash(Any_Key<K, V> type_key) {
		return typeid(Any_Key<K, V>).hash_code();
	}

public:

	template<class K, class V>
	bool contains(Any_Key<K ,V> key) const {
		return m_any_map.find(hash(key)) != m_any_map.end();
	}

	template<class K, class V>
	auto& operator [] (Any_Key<K, V> key) {
		static auto deleter = [](void* data) {
			delete reinterpret_cast<V*>(data);
		};

		if (!contains(key)) {
			void* data = reinterpret_cast<void*>(new V());
			m_any_map[hash(key)] = any_ptr(data, deleter);
		}

		return *reinterpret_cast<V*>(m_any_map[hash(key)].get());
	}

	template<class K, class V>
	auto& at (Any_Key<K, V> key) {
		return this->operator[](key);
	}


	template<class K, class V, class... DefaultArgs>
	auto& get(Any_Key<K, V> key, DefaultArgs&&... args) {
		if (!contains(key)) {
			return m_any_map[hash(key)] = V(std::forward(args)...);
		} else {
			return m_any_map[hash(key)];
		}
	}

	template<class K, class V>
	auto& get(Any_Key<K, V> key, V&& value) {
		if (!contains(key)) {
			return m_any_map[hash(key)] = value;
		} else {
			return m_any_map[hash(key)];
		}
	}

	template<class K, class V, class... Args>
	void emplace(Any_Key<K, V> key, Args&&... args) {
		//Todo should return pair<iterator, bool> of correct type
		m_any_map.emplace(hash(key), args...);
	}

	int empty() const { return m_any_map.empty(); }
	int size() const { return m_any_map.size(); }
	int max_size() const { return m_any_map.size(); }

	auto begin() const { return m_any_map.begin(); }
	auto end() const { return m_any_map.end(); }

	auto begin() { return m_any_map.begin(); }
	auto end() { return m_any_map.end(); }
	auto cbegin() { return m_any_map.cbegin(); }
	auto cend() { return m_any_map.cend(); }
};



}
}



#endif /* TYPEMAP_H_ */
