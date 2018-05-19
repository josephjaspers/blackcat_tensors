#include <string>
#include <vector>
#include <iostream>
//#include "../Structures/list_comprehension.h"
//#include "../Structures/omp_unique.h"
//#include "../Structures/binary_tree.h"
//#include "../Structures/forward_list.h"
//#include "../Structures/bidirectional_tuple.h"
//
//#include "../Structures/Parser/CSV_Parser.h"
//#include "../Structures/Parser/DataFrame.h"
#include "../Structures/static_hash_map.h"
#include "../Structures/stack_hash_map.h"
#include "../Structures/pthread_unique.h"
#include "../Structures/omp_unique.h"

//#include <pthread.h>
//#include <type_traits>


int main() {
//
//	//This tuple is better in some ways than the std::tuple -> supports forward/backward iteration, get is a built in method.
//	BC::Structure::Tuple<int, float, std::string> tuple(1, 3.12, std::string("third element"));
//	tuple.for_each([](auto& x) { std::cout << x << " "; } );
//
//	std::cout << std::endl << " printing 2nd " << tuple.get<1>();
//	std::cout << std::endl << std::endl;
//
//
//
//	BC::CSV::CSV_Parser parser;
////	parser.add_skip_rows(2);
////	parser.add_skip_cols(1);
//
//	parser.parse("///home/joseph///Downloads///parse_test.csv");
//
//	BC::CSV::DataFrame<int, std::string, std::string, float> df(parser);
//	df.ORDER_BY([](auto& x) { return x.head().data(); });
//
//
//	std::cout << std::endl << " now printing out df" << std::endl;
//	df.for_each([](auto& row) {
//		row.for_each([](auto& cell) {
//			std::cout << typeid(decltype(cell)).name() <<" = " << cell << " |";
//		});
//		std::cout << std::endl;
//
//	});
//
//
//	std::cout << std::endl << std::endl;
//
//
//	BC::Structure::binary_tree<int> tree;
//
//	tree.add(5);
//	tree.add(4);
//	tree.add(7);
//	tree.add(2);
//	tree.add(10);
//	tree.add(2);
//	tree.print();
//
//	std::cout << std::endl;
//	tree.remove(4);
//	tree.print();
//	tree.clear();
//	tree.print();




//	BC::Structure::static_hash_map<std::string, int> hmap(256);
	BC::Structure::stack_hash_map<4, std::string, int> hmap;

	hmap["cats"] = 4;
	hmap["dogs"] = 2;
	hmap["total_pets"] = 6;
	hmap["adgf"] = 1;
	hmap["total_petss"] = 7;


	hmap.print();



	std::cout << " success " << std::endl;
}
