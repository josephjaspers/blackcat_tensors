/*
 * CSV_Parser.h
 *
 *  Created on: Apr 28, 2018
 *      Author: joseph
 */

#ifndef CSV_PARSER_H_
#define CSV_PARSER_H_

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
namespace BC {
namespace CSV {

class CSV_Parser {

	using grid = std::vector<std::vector<std::string>>;

	std::string filename;
	std::ifstream is;
	grid data;

	std::vector<int> col_skips;
	std::vector<int> row_skips;
	std::vector<std::string> column_names;

public:

	template<class... integers>
	void add_skip_cols(int x, integers... ints) {
		col_skips.push_back(x);
		if constexpr (sizeof...(integers)!=0)
			add_skip_columns(ints...);
	}
	template<class... integers>
	void add_skip_rows(int x, integers... ints) {
		row_skips.push_back(x);
		if constexpr (sizeof...(integers)!=0)
			add_skip_rows(ints...);
	}

	CSV_Parser() = default;

	bool isSkipRow(int i ) {
		for (int row : row_skips) {
			if (row == i)
				return true;
		}

		return false;
	}
	bool isSkipCol(int i) {
		for (int row : col_skips) {
			if (row == i)
				return true;
		}

		return false;
	}

	void parse(std::string filename, bool skip_header = false, bool read_header = true) {
		is = std::ifstream(filename);
		std::vector<std::string> data_row;
		std::string row;
		std::string cell;
		int current_row = 0;
		int current_col = 0;



		column_names.clear();

		//IGNORE THE FIRST ROW IF SKIP HEADER
		if (skip_header)
			std::getline(is, row, '\n');

		//Else if we should read the header names to the file
		else if (read_header) {
			std::getline(is, row, '\n'); //this is the header line
			std::stringstream ss(row);

			while (ss.good()) {
				std::getline(ss, cell, ','); // get a header name

				if (!isSkipCol(current_col))
					column_names.push_back(cell);

				++current_col;
			}
		}


		//for each row -- read
		current_row = 0;
		while (is.good()) {
			std::getline(is, row, '\n');

			if (!is.good())
				return;

			// check if skip row, else for each line
			if (!isSkipRow(current_row)) {
				data_row = std::vector<std::string>(0);
				std::stringstream ss(row);

				current_col = 0;
				while (ss.good()) {
					std::getline(ss, cell, ',');
					//check is skip col, else read
					if (!isSkipCol(current_col))
						data_row.push_back(cell);

					++current_col;
				}
				data.push_back(data_row);
			}
			++current_row;
		}
	}

	const auto& getData() const {
		 return data;
	}
	void print() {
		for (std::string header : column_names) {
			std::cout << header << ", ";
		}
		std::cout << std::endl;

		for (int r = 0; r < data.size(); ++r) {
			for (int c = 0; c < data[r].size(); ++c) {
				std::cout << data[r][c] << ", ";
			}
			std::cout << std::endl;
		}
	}








};
}
}



#endif /* CSV_PARSER_H_ */
