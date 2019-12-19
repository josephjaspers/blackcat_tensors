/*
 * Layer_Cache.h
 *
 *  Created on: Aug 31, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_NEURALNETWORKS_LAYER_CACHE_H_
#define BLACKCATTENSORS_NEURALNETWORKS_LAYER_CACHE_H_

#include <vector>
#include <type_traits>

namespace BC {
namespace nn {

/**A type designed to act as a key to the Cache object.
 *
 * Arguments:
 * 	K - any class to use as a key, generally "Name<char...>" class is used to create a constexpr name to the class.
 * 	V - the type to return from the given key
 * 	IsRecurrent - Determines if the storing should override the most
 * 		recent member store or if it should be stored in a separate location for back-propagation through time.
 */
template<class K, class V, class IsRecurrent>
struct cache_key : BC::utility::Any_Key<K, V> {
	static_assert(
			std::is_same<IsRecurrent, std::true_type>::value ||
					std::is_same<IsRecurrent, std::false_type>::value,
			"Cache_Key `IsRecurrent` must be std::true_type or std::false_type");

	using is_recurrent = IsRecurrent;
};


/** A Dictionary designed to store any type using the 'store' and 'load' functions.
 *
 * The cache object stores any object that can be mapped from a unique cache_key.
 * Additionally the Cache object stores an integer time-index which determines the current
 * 'time' to return when loading a Value from a Recurrent key.
 *
 * The time_index is only relevant with recurrent keys.
 */
struct Cache {

	template<class K, class V, class R>
	using key_type = cache_key<K, V, R>;

	int m_time_index = 0;
	BC::utility::Any_Map cache;

private:

	template<class K, class V>
	auto hash(key_type<K, V, std::true_type> key) {
		return BC::utility::Any_Key<K, std::vector<V>>();
	}

	template<class K, class V>
	auto hash(key_type<K, V, std::false_type> key) {
		return BC::utility::Any_Key<K, V>();
	}

public:

	template<class K, class V, class R>
	bool contains(key_type<K,V,R> key) {
		return cache.contains(key);
	}

	///Loads the current value at the current time_index
	template<class K, class V>
	auto& load(key_type<K, V, std::true_type> key, int t_modifier=0) {
		std::vector<V>& history = cache[hash(key)];
		int index = history.size()- 1 - m_time_index + t_modifier;

		BC_ASSERT(index < history.size(),
			"Load recurrent_variable index out of bounds"
				"\nHistory size: " + std::to_string(history.size()) +
				"\nIndex:" + std::to_string(index));

		return history[index];
	}

	template<class K, class V>
	auto& load(key_type<K, V, std::false_type> key, int t_modifier=0) {
		BC_ASSERT(t_modifier==0, "Nonrecurrent keys cannot have a time_offset");
		return cache[hash(key)];
	}

	template<class K, class V, class DefaultFactory>
	auto& load(key_type<K, V, std::true_type> key,
			int t_modifier,
			DefaultFactory function) {

		std::vector<V>& history = cache[hash(key)];

		unsigned index = history.size()- 1 - m_time_index + t_modifier;
		if (index >= history.size()) {
			history.push_back(function());
			return history.back();
		}

		BC_ASSERT(index < history.size(),
			"Load recurrent_variable index out of bounds"
				"\nHistory size: " + std::to_string(history.size()) +
				"\nIndex:" + std::to_string(index));

		return history[index];
	}

	template<class K, class V, class DefaultFactory>
	auto& load(key_type<K, V, std::true_type> key, DefaultFactory function) {
		return load(key, 0, function);
	}


	template<class K, class V, class DefaultFactory>
	auto& load(key_type<K, V, std::false_type> key, DefaultFactory function) {
		auto hkey = hash(key);

		if (cache.contains(hkey)) {
			return cache[hkey];
		} else {
			return cache[hkey] = function();
		}
	}

	template<class K, class V, class U>
	auto& store(key_type<K, V, std::true_type> key, U&& expression) {
		cache[hash(key)].push_back(std::forward<U>(expression));
		return cache[hash(key)].back();
	}

	template<class K, class V, class U>
	auto& store(key_type<K, V, std::false_type> key, U&& expression) {
		if (cache.contains(hash(key))) {
			return cache[hash(key)] = std::forward<U>(expression);
		} else {
			return cache[hash(key)] = V(std::forward<U>(expression));
		}
	}

	int get_time_index() const { return m_time_index; }
	void increment_time_index() { m_time_index++; }
	void decrement_time_index() { m_time_index--; }
	void zero_time_index() { set_time_index(0); }
	void set_time_index(int idx) { m_time_index = idx; }

	template<class K, class V>
	void clear_bp_storage(key_type<K, V, std::false_type> key) {}

	template<class K, class V>
	void clear_bp_storage(key_type<K, V, std::true_type> key) {
		auto& storage = cache[hash(key)];

		if (storage.size() > 1) {
			auto last = std::move(storage.back());
			storage.clear();
			storage.push_back(std::move(last));
		}
	}
};


}
}


#endif /* LAYER_CACHE_H_ */
