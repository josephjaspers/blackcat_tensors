/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_IO_H_
#define BLACKCAT_IO_H_

#include "BlackCat_Common.h"
#include "BlackCat_Tensors.h"
#include "BlackCat_String.h"
#include <algorithm>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <assert.h>

namespace BC {
namespace io {

template<class T>
static T from_string(const std::string& str);

#define from_string_def(dtype, ...)\
template<>\
inline dtype from_string(const std::string& str) {\
	return __VA_ARGS__;\
}
from_string_def(double, std::stod(str))
from_string_def(float, std::stof(str))
from_string_def(int, std::stoi(str))
from_string_def(std::string, str)

template<class T>
struct Range {
	T begin_;
	T end_;
	T begin() const { return begin_; }
	T end()   const { return end_; }
};
template<class T>
auto range(T begin, T end=T()) {
	return Range<T>{begin, end};
}

struct csv_descriptor {

#define FORWARDED_PARAM(dtype, name, default_value)	\
dtype name##_ = default_value;											\
csv_descriptor& name(dtype name) {	\
	name##_ = name; 				\
	return *this; 					\
}									\
const dtype& name() const { 		\
	return name##_; 				\
}									\

	csv_descriptor(std::string fname) : filename_(fname) {}

	FORWARDED_PARAM(std::string, filename, "")
	FORWARDED_PARAM(bool, header, true)
	FORWARDED_PARAM(bool, index, false)
	FORWARDED_PARAM(char, mode, 'r')
	FORWARDED_PARAM(char, delim, ',')
	FORWARDED_PARAM(char, row_delim, '\n')
	FORWARDED_PARAM(bool, transpose, false)

	FORWARDED_PARAM(std::vector<int>, skip_rows, {})
	FORWARDED_PARAM(std::vector<int>, skip_cols, {})

	template<class... Integers>
	csv_descriptor& skip_rows(int x, Integers... args_) {
		skip_rows_ = std::vector<int> {x, args_...};
		return *this;
	}

	template<class... Integers>
	csv_descriptor& skip_cols(int x, Integers... args_) {
		skip_cols_ = std::vector<int> {x, args_...};
		return *this;
	}

};

static std::vector<std::vector<BC::string>> parse(csv_descriptor desc)
{
	using BC::string;
	using std::vector;

	std::ifstream ifs(desc.filename());

	auto find = [](auto& collection, auto var) -> bool {
		return std::find(
				collection.begin(),
				collection.end(),
				var) != collection.end();
	};

	if (!ifs.good()) {
		BC::print("Unable to open `", desc.filename(), '`');
		throw 1;
	}

	string csv_text = string(std::istreambuf_iterator<char>(ifs),
			std::istreambuf_iterator<char>());
	vector<string> rows = csv_text.split(desc.row_delim());
	vector<vector<string>> split_rows;

	int curr_col = 0;

	for (string& row : rows) {
		auto cells = row.split(desc.delim());

		if (!split_rows.empty() &&
				split_rows.back().size() != cells.size()) {
			BC::print("Column length mismatch."
					"\nExpected: ",  split_rows.back().size(),
					"\nReceived: ",  cells.size(),
					"\nRow index: ", split_rows.size());
			throw 1;
		}

		if (!cells.empty() && !find(desc.skip_rows(), curr_col)) {
			if (desc.skip_cols().empty())
				split_rows.push_back(cells);
			else {
				vector<string> curr_row;
				for (std::size_t i = 0; i < cells.size(); ++i) {
					if (!find(desc.skip_cols(), i))
						curr_row.push_back(std::move(cells[i]));
				}
				split_rows.push_back(curr_row);
			}
		}
		++curr_col;
	}

	return split_rows;
}

template<
		class ValueType,
		class Allocator=BC::Allocator<BC::host_tag, ValueType>>
static BC::Matrix<ValueType, Allocator> read_uniform(
		csv_descriptor desc,
		Allocator alloc=Allocator()) {

	using BC::string;
	using std::vector;


	if (desc.transpose()){
		BC::print("Transpose is not supported for read_uniform");
		BC::print("TODO implement transposition");
		throw 1;
	}

	vector<vector<string>> data = parse(desc);

	int rows = data.size() - desc.header();
	int cols = data[0].size() - desc.index();

	if (desc.transpose())
		std::swap(rows, cols);

	BC::Matrix<ValueType, Allocator> matrix(rows, cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			int d_i = i + desc.header();
			int d_j = j + desc.index();

			if (desc.transpose())
				matrix[i][j] = from_string<ValueType>(data[d_i][d_j]);
			else
				matrix[j][i] = from_string<ValueType>(data[d_i][d_j]);
		}
	}

	return matrix;
}

}
}


#endif /* BLACKCAT_IO_H_ */
