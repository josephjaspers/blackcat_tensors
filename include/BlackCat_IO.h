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
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
namespace BC {

template<class T>
struct range {
	T begin;
	T end = T();
};

struct csv {

	const char* filename;
	bool header_ = true;
	bool index_ = false;
	char delim_ = ',';

	csv& header(bool header) {
		header_ = header;
		return *this;
	}
	csv& index(bool index) {
		index_ = index;
		return *this;
	}
	csv& delim(bool delim) {
		delim_ = delim;
		return *this;
	}


};


template<class ValueType>
static BC::Matrix<ValueType, Allocator=BC::Allocator<BC::host_tag, ValueType>>
read_uniform(csv csv) {

	  std::ifstream ifs(csv.filename);
	  std::stringstream str_buf;
	  std::vector<std::string> row_buf;

	  using iter = std::istreambuf_iterator<char>;
	  rows = std::count_if(iter(ifs), iter(),
			  [=](char c) { return c == csv.delim; },
			  [](char c) { return c == '\n'; });

	  for (char c : range {std::istreambuf_iterator<char>(ifs)}) {
		  if (c == csv.delim) {
			  row_buf.push_back(str_bufZ);
		  }
	  }
}





}


#endif /* BLACKCAT_IO_H_ */
