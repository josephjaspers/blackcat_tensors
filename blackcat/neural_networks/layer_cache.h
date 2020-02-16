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

namespace bc {
namespace nn {

/**A type designed to act as a key to the Cache object.
 *
 * Arguments:
 * 	K - any class to use as a key, generally "Name<char...>" class is used to create a constexpr name to the class.
 * 	V - the type to return from the given key
 * 	IsRecurrent - Determines if the storing should override the most
 * 		recent member store or if it should be stored in a separate location for back-propagation through time.
 */

enum cache_key_type {
	inherit,
	always_recurrent,
	always_forward
};

template<class K, class V, cache_key_type CacheKeyOverrider=inherit>
struct cache_key : bc::utility::Any_Key<K, V> {
	static constexpr cache_key_type cache_override_type = CacheKeyOverrider;
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

	template<class K, class V, cache_key_type R>
	using key_type = cache_key<K, V, R>;

private:

	int m_time_index = 0;
	bool is_recurrent = false;

	bc::utility::Any_Map cache;

	template<class K, class V>
	auto hash(key_type<K, V, cache_key_type::always_recurrent> key) {
		return bc::utility::Any_Key<K, std::vector<V>>();
	}

	template<class K, class V, cache_key_type R>
	auto hash(key_type<K, V, R> key) {
		return bc::utility::Any_Key<K, V>();
	}

public:

	void enable_recurrent_caching(bool enable=true) {
		is_recurrent = enable;
	}

	template<class K, class V, cache_key_type R>
	bool contains(key_type<K,V,R> key) {
		return cache.contains(key);
	}

	template<class K, class V>
	auto& load(key_type<K, V, cache_key_type::inherit> key, int t_modifier=0)
	{
		if (is_recurrent)
			return load(key_type<K,V, always_recurrent>(), t_modifier);
		else
			return load(key_type<K,V, always_forward>(), t_modifier);
	}

	template<class K, class V, class Factory>
	auto& load(key_type<K, V, cache_key_type::inherit> key, int t_modifier, Factory factory)
	{
		if (is_recurrent)
			return load(key_type<K,V, cache_key_type::always_recurrent>(), t_modifier, factory);
		else
			return load(key_type<K,V, cache_key_type::always_forward>(), factory);
	}

	template<class K, class V, class Factory>
	auto& load(key_type<K, V, cache_key_type::inherit> key, Factory factory)
	{
		return load(key, 0, factory);
	}

	template<class K, class V, class U>
	auto& store(key_type<K, V, cache_key_type::inherit> key, U&& expression) {
		if (is_recurrent)
			return store(key_type<K,V, cache_key_type::always_recurrent>(), expression);
		else
			return store(key_type<K,V, cache_key_type::always_forward>(), expression);
	}

	///Loads the current value at the current time_index
	template<class K, class V>
	auto& load(key_type<K, V, cache_key_type::always_recurrent> key, int t_modifier=0) {
		std::vector<V>& history = cache[hash(key)];
		unsigned index = history.size()- 1 - m_time_index + t_modifier;

		BC_ASSERT((int)index < (int)history.size(),
			"Load recurrent_variable index out of bounds"
				"\nHistory size: " + std::to_string(history.size()) +
				"\nIndex:" + std::to_string(index));

		return history[index];
	}

	template<class K, class V>
	auto& load(key_type<K, V, cache_key_type::always_forward> key, int t_modifier=0) {
		BC_ASSERT(t_modifier==0, "Nonrecurrent keys cannot have a time_offset");
		return cache[hash(key)];
	}

	template<class K, class V, class DefaultFactory>
	auto& load(key_type<K, V, cache_key_type::always_recurrent> key,
			int t_modifier,
			DefaultFactory function)
	{
		std::vector<V>& history = cache[hash(key)];

		unsigned index = history.size()- 1 - m_time_index + t_modifier;
		if (index >= history.size()) {
			history.push_back(function());
			return history.back();
		}

		BC_ASSERT((int)index < (int)history.size(),
			"Load recurrent_variable index out of bounds"
				"\nHistory size: " + std::to_string(history.size()) +
				"\nIndex:" + std::to_string(index));

		return history[index];
	}

	template<class K, class V, class DefaultFactory>
	auto& load(key_type<K, V, cache_key_type::always_recurrent> key, DefaultFactory function) {
		return load(key, 0, function);
	}


	template<class K, class V, class DefaultFactory>
	auto& load(key_type<K, V, cache_key_type::always_forward> key, DefaultFactory function) {
		auto hkey = hash(key);

		if (cache.contains(hkey)) {
			return cache[hkey];
		} else {
			return cache[hkey] = function();
		}
	}

	template<class K, class V, class U>
	auto& store(key_type<K, V, cache_key_type::always_recurrent> key, U&& expression) {
		cache[hash(key)].push_back(std::forward<U>(expression));
		return cache[hash(key)].back();
	}

	template<class K, class V, class U>
	auto& store(key_type<K, V, cache_key_type::always_forward> key, U&& expression) {
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
	void clear_bp_storage(key_type<K, V, cache_key_type::always_forward> key) {}

	template<class K, class V>
	void clear_bp_storage(key_type<K, V, cache_key_type::inherit> key)
	{
		if (is_recurrent) {
			auto k = key_type<K,V, cache_key_type::always_recurrent>();
			clear_bp_storage(k);
		}
	}

	template<class K, class V>
	void clear_bp_storage(key_type<K, V, cache_key_type::always_recurrent> key) {
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
