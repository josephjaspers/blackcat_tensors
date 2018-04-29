#include <string>
#include <vector>
#include <iostream>
#include "../Structures/list_comprehension.h"
#include "../Structures/omp_unique.h"
#include "../Structures/binary_tree.h"
#include "../Structures/forward_list.h"
#include "../Structures/hash_map.h"
#include "../Structures/bidirectional_tuple.h"

#include "../Structures/Parser/CSV_Parser.h"
#include "../Structures/Parser/DataFrame.h"

#include <pthread.h>
#include <type_traits>

struct hasher{
	int operator()(int i)  const {
		return i % 7;
	}
};

int main() {

	//This tuple is better in some ways than the std::tuple -> supports forward/backward iteration, get is a built in method.
	BC::Structure::Tuple<int, float, std::string> tuple(1, 3.12, std::string("third element"));
	tuple.for_each([](auto& x) { std::cout << x << " "; } );

	std::cout << std::endl << " printing 2nd " << tuple.get<1>();
	std::cout << std::endl << std::endl;



	BC::CSV::CSV_Parser parser;
//	parser.add_skip_rows(2);
//	parser.add_skip_cols(1);

	parser.parse("///home/joseph///Downloads///parse_test.csv");
	parser.print();

	BC::CSV::DataFrame<int, std::string, std::string, float> df(parser);


	std::cout << std::endl << " now printing out df" << std::endl;
	df.for_each([](auto& row) {
		row.for_each([](auto& cell) {
			std::cout << typeid(decltype(cell)).name() <<" = " << cell << " |";
		});
		std::cout << std::endl;

	});


	std::cout << std::endl << std::endl;


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
