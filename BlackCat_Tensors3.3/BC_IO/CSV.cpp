#include "CSV.h"

namespace BC {
namespace IO {

CSV::CSV(std::string filename_, bool header_present) :
		filename(filename_),
		is(filename_),
		no_header(!header_present) {

	parse_header();
}

int CSV::header_index(std::string name) const {
	if (no_header)
		throw std::invalid_argument("no header utilized");

	for (int i = 0; i < headers.size(); ++i) {
		if (name == headers[i])
			return i;
	}
	throw std::invalid_argument("no header found for ignore header");
}

void CSV::parse_header() {
	headers.clear();
	std::string header_row;
	std::string header_col;

	if (no_header)
		return;

	std::getline(is, header_row, '\n'); //this is the header line
	std::stringstream header_ss(header_row);

	while (header_ss.good()) {
		std::getline(header_ss, header_col, column_delimeter); // get a header name
		headers.push_back(header_col);
	}
}
void CSV::parse() {
	data.clear();
	std::vector<std::string> data_row;
	std::string row;
	std::string cell;
	int current_row = 0;
	int current_col = 0;

	current_row = 0;
	while (is.good()) {
		std::getline(is, row, '\n');

		if (!is.good())
			return;

		// check if skip row, else for each line
		if (!is_ignr_row(current_row)) {
			data_row = std::vector<std::string>(0);
			std::stringstream ss(row);

			current_col = 0;
			while (ss.good()) {
				std::getline(ss, cell, ',');
				//check is skip col, else read
				if (!is_ignr_col(current_col))
					data_row.push_back(cell);

				++current_col;
			}
			data.push_back(data_row);
		}
		++current_row;
	}
}
void CSV::parse_next(int lines) {
	data.clear();
	std::vector<std::string> data_row;
	std::string row;
	std::string cell;
	int current_row = 0;
	int current_col = 0;

	current_row = 0;
	while (is.good() && lines > 0) {
		std::getline(is, row, '\n');

		if (!is.good())
			return;

		// check if skip row, else for each line
		if (!is_ignr_row(current_row)) {
			data_row = std::vector<std::string>(0);
			std::stringstream ss(row);

			current_col = 0;
			while (ss.good()) {
				std::getline(ss, cell, ',');
				//check is skip col, else read
				if (!is_ignr_col(current_col))
					data_row.push_back(cell);

				++current_col;
			}
			data.push_back(data_row);
		}
		++current_row;
		--lines;
	}
}

void CSV::print() {
	if (!no_header)
	for (std::string header : headers) {
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
}
}
