/*
 * BlackCat_IO.h
 *
 *  Created on: May 29, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_IO_H_
#define BLACKCAT_IO_H_

#include "BlackCat_Common.h"
#include "BlackCat_Tensors.h"
#include <algorithm>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <assert.h>

namespace BC {

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

	std::string filename;
	char mode_ = 'r';
	bool header_ = true;
	bool index_ = false;
	char delim_ = ',';
	bool transpose_ = false;
	std::vector<int> skip_rows_;
	std::vector<int> skip_cols_;

	csv_descriptor(std::string filename_, char mode  = 'r'):
		filename(filename_),
		mode_(mode) {}

	csv_descriptor& header(bool header) {
		header_ = header;
		return *this;
	}
	csv_descriptor& index(bool index) {
		index_ = index;
		return *this;
	}
	csv_descriptor& delim(char delim) {
		delim_ = delim;
		return *this;
	}
	csv_descriptor& mode(bool mode) {
		assert(mode == 'r' || mode == 'w');
		mode_ = mode;
		return *this;
	}
	csv_descriptor& transpose(bool transpose) {
		transpose_ = transpose;
		return *this;
	}

	csv_descriptor& skip_rows(std::vector<int> srows) {
		skip_rows_ = srows;
		return *this;
	}
	csv_descriptor& skip_cols(std::vector<int> scols) {
		skip_cols_ = scols;
		return *this;
	}
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

	const std::vector<int>& skip_rows() const {
		return skip_rows_;
	}
	const std::vector<int>& skip_cols() const {
		return skip_cols_;
	}

	bool transpose() const {
		return transpose_;
	}
	bool header() const {
		return header_;
	}
	bool index() const {
		return index_;
	}
	char delim() const {
		return delim_;
	}
	bool mode() const {
		return mode_;
	}


};


template<class ValueType, class Allocator=BC::Allocator<BC::host_tag, ValueType>>
static BC::Matrix<ValueType, Allocator> read_uniform(csv_descriptor csv, Allocator alloc=Allocator()) {

	  std::ifstream ifs(csv.filename);
	  std::stringstream str_buf;
	  std::vector<ValueType> cell_buf;

	  bool header_skipped = !csv.header();
	  bool cols_counted = false;
	  int n_cols = 0;
	  int curr_col = 0;
	  int curr_col_with_skips = 0;

	  int n_rows = 0;
	  int curr_row_with_skips = 0;

	  for (char c : range(std::istreambuf_iterator<char>(ifs))) {
		  if (std::find(csv.skip_rows().begin(),
				  	  	  csv.skip_rows().end(), curr_row_with_skips) != csv.skip_rows().end()) {
			  if (c == '\n')
				  curr_row_with_skips++;
			  continue;
		  }

		  if (std::find(csv.skip_cols().begin(),
				  	  	 csv.skip_cols().end(), curr_col_with_skips) != csv.skip_cols().end()) {
			  if (c == csv.delim())
				  curr_col_with_skips++;
			  continue;
		  }

		  if (c == '\n') {
			  if (!cols_counted) {
				  n_cols++;
				  cols_counted=true;
			  }
			  curr_col++;
			  curr_col_with_skips++;

			  if (header_skipped){
				  n_rows++;
				  curr_row_with_skips++;
				  cell_buf.push_back(from_string<ValueType>(str_buf.str()));
				  str_buf.str("");
				  str_buf.clear();
			  } else {
				  header_skipped = true;
			  }

			  BC_ASSERT(curr_col==n_cols, "Invalid row found on " + std::to_string(n_rows));
			  curr_col=0;
			  curr_col_with_skips=0;
		  } else if (c == csv.delim()) {
			  if (!cols_counted) {
				  n_cols++;
			  }
			  curr_col++;
			  curr_col_with_skips++;

			  if (header_skipped) {
				  cell_buf.push_back(from_string<ValueType>(str_buf.str()));
			  }
			  str_buf.str("");
			  str_buf.clear();
		  }
		  else if (header_skipped) {
			  str_buf << c;
		  }
	  }

	  if (str_buf.str() != "") {
		  n_rows++;
		  curr_row_with_skips++;

		  cell_buf.push_back(from_string<ValueType>(str_buf.str()));
		  str_buf.str("");
		  str_buf.clear();
	  }

	  BC::Matrix<ValueType> data(
			  n_cols, n_rows);

	  std::copy(cell_buf.begin(), cell_buf.end(), data.begin());

	  if (csv.transpose()) {
		  BC::Matrix<ValueType, Allocator> data_t(BC::make_shape(n_cols, n_rows), alloc);
		  data_t.copy(data);
		  return data_t;
	  } else {
		  BC::Matrix<ValueType> data_transposed(n_rows, n_cols);
		  data_transposed = data.transpose();

		  BC::Matrix<ValueType, Allocator> data_correct_system(BC::make_shape(n_rows, n_cols), alloc);
		  data_correct_system.copy(data_transposed);
		  return data_correct_system;
	  }
}




}


#endif /* BLACKCAT_IO_H_ */
