
#ifndef CONCURRENT_HASH_MAP_H_
#define CONCURRENT_HASH_MAP_H_


namespace BC {
namespace Structure {

template<class K, class V, class hasher, class deleter = default_deleter>
struct concurrent_hash_map {

	struct data {
		V value;
		K key;
	};


	const deleter destroy 	= deleter();
	const hasher hash		= hasher();

	int bucket_size;
};
}
}



#endif /* CONCURRENT_HASH_MAP_H_ */
