/*
 * stack_hash_map.h
 *
 *  Created on: May 19, 2018
 *      Author: joseph
 */

#ifndef STACK_HASH_MAP_H_
#define STACK_HASH_MAP_H_

#include "BC_Collections.h"
/*
 * None rescaling hashmap.
 * Once the hashmap reaches maximum capacity it simply begins to re-write over older entries, not thread safe.
 *
 */

namespace BC {
namespace Structure {

template<int SIZE, class K, class V, class hasher = std::hash<K>>
struct  stack_hash_map{

	const hasher hash;
	static constexpr int bucket_length = SIZE;

	struct data {
		bool initialized = false;

		K key;
		V value;

		data() = default;
		data(K key_, V value_ = V())
			: key(key_),
			  value(value_),
			  initialized(true)
		{
			/*empty*/
		}

		//get a simple string format of the data
		std::string str() const {
			return std::string("Key: " + to_str(key) + ", Value: " + to_str(value));
		}

	};

	stack_hash_map(hasher hash_ = hasher())
		: hash(hash_) {}

	int element_count = 0;
	data bucket[bucket_length];

	int size() const {
		return element_count;
	}

	bool full() const {
		return element_count == bucket_length;
	}

	V& full_return(const K& key, int index) {
		int t_index = index;

		while (hash(bucket[t_index].key) % bucket_length == (unsigned)index) {
			if (bucket[t_index].key == key)
				return bucket[t_index].value;

			t_index++;
		}
		//initialize element resets the element
		initialize_element(key, index);
		return bucket[index].value;
	}


	V& operator [] (const K& key) {
		int index = hash(key) % bucket_length;

		if (full())
			return (full_return(key, index));

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

	void initialize_element(const K& key, int index) {
		bucket[index].initialized = true;
		bucket[index].key = key;
		element_count++;
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

	void print() {
		int index = 0;
		while (index < bucket_length) {
			if (bucket[index].initialized)
				std::cout << "{ Index: " << index << ", "<< bucket[index].str() << " }" << std::endl;
			index++;
		}
	}
};


}
}



#endif /* STACK_HASH_MAP_H_ */
