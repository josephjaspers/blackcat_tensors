#include <string>
#include <vector>
#include <iostream>
#include "../Structures/list_comprehension.h"
#include "../Structures/omp_unique.h"
#include "../Structures/binary_tree.h"
#include "../Structures/forward_list.h"
#include "../Structures/hash_map.h"
#include <pthread.h>

struct hasher{
	int operator()(int i)  const {
		return i % 7;
	}
};

int main() {

	BC::Structure::binary_tree<int> tree;

	tree.add(5);
	tree.add(4);
	tree.add(7);
	tree.add(2);
	tree.add(10);
	tree.add(2);
	tree.print();

	std::cout << std::endl;
	tree.remove(4);
	tree.print();
	tree.clear();
	tree.print();




//	BC::Structure::hash_map<int, int, hasher> hmap;
//	hmap[9] = 4;
////	hmap[2] = 1;

//	std::cout << hmap[2] << std::endl;


	std::cout << " success " << std::endl;
}
