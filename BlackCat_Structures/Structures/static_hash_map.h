/*
 * hash_map.h
 *
 *  Created on: May 18, 2018
 *      Author: joseph
 */

#ifndef STATIC_HASH_MAP_H_
#define STATIC_HASH_MAP_H_

#include "BC_Collections.h"
#include <functional>

//Non rescaling hash_map



namespace BC{
namespace Structure {

template<class K, class V, class hasher = std::hash<K>, class deleter = default_deleter>
struct static_hash_map {

	static constexpr hasher hash = hasher();
	static constexpr int default_size = 256;

	struct internal {
		bool initialized = false;

		K key;
		V value;

		internal() = default;
		internal(K key_, V value_ = V())
			: key(key_),
			  value(value_),
			  initialized(true)
		{
			/*empty*/
		}

		internal(const internal&) = default;
		internal(internal&&) = default;

		//get a simple string format of the internal
		std::string str() const {
			return std::string("Key: " + to_str(key) + ", Value: " + to_str(value));
		}
	};

	int bucket_length;
	int element_count;

	//internal internal array
	internal* bucket;

	static_hash_map(int default_sz = default_size) {
		bucket = new internal[default_sz];
		bucket_length = default_sz;
		element_count = 0;
	}

	int size() const {
		return element_count;
	}

	V& operator [] (const K& key) {
		int index = hash(key) % bucket_length;

		//first fount perfect O(1)
		if (bucket[index].key == key) {
			return bucket[index].value;
		} else {
			return linear_probe(key, index);
		}
	}

	V* contains(const K& key) {
		int index = hash(key) % bucket_length;

		//first fount perfect O(1)
		if (bucket[index].key == key)
			return &bucket[index].value;
		else
			return safe_linear_probe(key, index);
	}

	void print() {
		int index = 0;
		while (index < bucket_length) {
			if (bucket[index].initialized)
				std::cout << "{ Index: " << index << ", "<< bucket[index].str() << " }" << std::endl;
			index++;
		}
	}

	void initialize_element(const K& key, int index) {
		bucket[index].initialized = true;
		bucket[index].key = key;
		element_count++;

		assert_range_bounds();
	}

	V& linear_probe(const K& key, int index) {
		//search
		while (bucket[index].initialized){
			if (bucket[index].key == key)
				return bucket[index].value;

			index = (index + 1) % bucket_length;
		}

		//if not found
		initialize_element(key, index);
		return bucket[index].value;
	}

	//same as linear_probe but doesn't initialize an element if not placed
	V* safe_linear_probe(const K& key, int index) {
		//search
		while (bucket[index].initialized){
			if (bucket[index].key == key)
				return &bucket[index].value;

			index = (index + 1) % bucket_length;
		}
		return nullptr;
	}


	void assert_range_bounds() {
		if (element_count >= bucket_length)
			throw std::invalid_argument("static_hash_map size capacity maxed");
	}


};

}
}


#endif /* HASH_MAP_H_ */
