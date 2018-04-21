///*
// * hash_map.h
// *
// *  Created on: Apr 18, 2018
// *      Author: joseph
// */
//
//#ifndef HASH_MAP_H_
//#define HASH_MAP_H_
//
//#include "BC_Collections.h"
//#include "forward_list.h"
//namespace BC {
//namespace Structure {
//
//template<class K, class V, class hasher, class deleter = default_deleter>
//struct hash_map : Collection<V, deleter> {
//
//	struct data {
//		V value;
//		K key;
//		bool initialized = false;
//
//		data() = default;
//		data(K k, V v=V()) :
//				key(k), value(v), initialized(true) {
//		}
//		bool operator == (const data& comp){
//			return (key == comp.key) && (initialized == comp.initialized);
//		}
//	};
//	using bucket_type = forward_list<data, deleter>;
//
//	mutable bucket_type* bucket;
//	const hasher hash = hasher();
//	const deleter del = deleter();
//
//	mutable int sz;
//	mutable int bucket_size = 12;
//	float load_factor = .75f;
//	float rescale_factor = 1.5f;
//
//	void clear() {
//		for (int i = 0; i < bucket_size; ++i) {
//			bucket[i].clear();
//		}
//		sz = 0;
//	}
//
//	int index (K key) {
//		return hash(key) % bucket_size;
//	}
//	bool empty() const override{
//		return !sz;
//	}
//	int size() const override {
//		return sz;
//	}
//	void update() {
//		if (sz > bucket_size * load_factor) {
//
//			int old_sz = bucket_size;
//			bucket_size *= rescale_factor;
//			bucket_type* n_bucket = new bucket_type[bucket_size];
//
//			for (int i = 0; i < old_sz; ++i)
//				if (!bucket[i].empty())
//					n_bucket[index(bucket[i].front().key)] = std::move(bucket[i]);
//
//			for (int i = 0; i < old_sz; ++i)
//				bucket[i].clear();
//
//			//reassign
//			bucket = n_bucket;
//		}
//	}
//
//	V& operator [](K key) {
//
//		std::cout << "op";
//		int index = hash(key) % bucket_size;
//		std::cout << " indx = " << index;
//
//		std::cout << std::endl;
//
//		if (!bucket[index].contains(data(key))) {
//			bucket[index].add((data(key)));
//			std::cout << "Adding " << std::endl;
//			sz++;
//			update();
//			std::cout << "returning " << std::endl;
//
//			return bucket[index].get(data(key))->value;
//
//		} else  {
//			return bucket[index].get(data(key))->value;
//		}
//	}
//
//	bool add(K key) override {
//		throw std::invalid_argument("add not defined for hashmap -- use operator[]");
//	}
//
//};
//
//}
//}
//#endif /* HASH_MAP_H_ */
