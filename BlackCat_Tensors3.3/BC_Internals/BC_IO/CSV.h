/*
 * Converts a CSV into a 2d vector of strings.
 * 			-> Trivial implementation,
 * 			   Supports column/row ignoring
 * 			   Meant to specifically work with DataFrame object
 */

#ifndef CSV_PARSER_H_
#define CSV_PARSER_H_

#include <set>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

namespace BC  {
namespace IO {

class CSV {
	using grid = std::vector<std::vector<std::string>>;

	std::string filename;			//name of file being read
	std::ifstream is;				//the input stream
	grid data;						//the internal data_frame

	bool no_header = false;				//boolean determining if there is no header in the file
	std::vector<std::string> headers;	//contains all the header names (including skip columns)

	std::set<int> ignr_cs;	//columns to ignore
	std::set<int> ignr_rs;			//rows    to ignore

	std::set<char> ignr_chars		= { ' ', '\t' };
	char column_delimeter 			= ',';
	char row_delimeter 				= '\n';

public:

	CSV(std::string filename, bool header_present = true);
private:
	int header_index(std::string) const;
public:

	template<class... strs>
	void ignr_cols(std::string s, strs... cols) { ignr_cs.insert(header_index(s));  ignr_cols(cols...); }
	void ignr_cols(std::string s) 				{ ignr_cs.insert(header_index(s)); }

	template<class... integers>
	void ignr_cols(int x, integers... ints)   { ignr_cs.insert(x);  ignr_cs_indexes(ints...); }
	void ignr_cols(int x) 					  { ignr_cs.insert(x); }

	template<class... integers>
	void ignr_rows(int x, integers... ints) { ignr_rs.insert(x);  ignr_rows(ints...); }
	void ignr_rows(int x) 					{ ignr_rs.insert(x); }

private:
	bool is_ignr_col(std::string s) const { return ignr_cs.find(header_index(s)) != ignr_cs.end(); }
	bool is_ignr_col(int i) 		const { return ignr_cs.find(i) != ignr_cs.end(); }
	bool is_ignr_row(int i) 		const { return ignr_rs.find(i) != ignr_rs.end(); }


	void parse_header();



public:
	void parse();
	void parse_next(int lines);
	void print();
	const auto& get_data() const { return data; }
	auto& get_data() { return data; }

};
}
}



#endif /* CSV_PARSER_H_ */
