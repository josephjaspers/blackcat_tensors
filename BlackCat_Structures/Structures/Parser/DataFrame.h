/*
 * Wednesday May 2nd
 *
 * DataFrame objects accepts a parsed CSV file (from CSV_Parser.h)
 * and converts it into a 2d matrix of (rows by columns) in which the columns
 * of the data match the given type.
 *
 * Current work: involves adding the ability to override the default parsing of a string -> type
 * 				 This is to enable users to parse data into a dataframe aside from the given BC_from_string
 *
 *
 */

#ifndef DATAFRAME_H_
#define DATAFRAME_H_

#include "CSV_Parser.h"
#include "../bidirectional_tuple.h"
#include <vector>
#include <algorithm>

namespace BC {
namespace CSV {

template<class T> T BC_from_string(std::string str);
template<> int BC_from_string<int>(std::string str) { return stoi(str); }
template<> float BC_from_string<float>(std::string str) { return stof(str); }
template<> double BC_from_string<double>(std::string str) { return stod(str); }
template<> long BC_from_string<long>(std::string str) { return stol(str); }
template<> unsigned BC_from_string<unsigned>(std::string str) { return stoi(str); }
template<> char BC_from_string<char>(std::string str) { return (str[0]); }

template<> std::string BC_from_string<std::string>(std::string str) { return (str); }

template<class... Ts>
class DataFrame {

	using row = Structure::Tuple<Ts...>;
	using grid = std::vector<row>;

	grid dataframe;

	std::vector<int> skip_columns;
	std::vector<int> skip_rows;

public:


	template<class comp>
	void ORDER_BY(comp c) {
		std::sort(dataframe.begin(), dataframe.end(), [&](auto& x, auto& y) { return c(x) > c(y); });
	}

	auto& getGrid() {
		return dataframe;
	}

	template<class Node>
	void generateRowData(Node& n, std::vector<std::string> row_data, int i) const {
		n.data() = BC_from_string<typename Node::type>(row_data[i]);

		if constexpr (n.hasNext())
				generateRowData(n.next(), row_data, i + 1);
	}

	auto generateRowData(std::vector<std::string> row_data) {
		row r;
		generateRowData(r.head(), row_data, 0);
		return r;
	}

	DataFrame(const CSV_Parser& csv) : dataframe(csv.getData().size()) {
		const auto& grid = csv.getData();

		for (int i = 0; i < grid.size(); ++i) {
			dataframe[i] = generateRowData(grid[i]);
		}
	}

	template<class functor>
	void for_each(functor f) {
		for (row& r : dataframe) {
			f(r);
		}
	}


};


}}

#endif /* DATAFRAME_H_ */
