/*
 * Layer_Loader.h
 *
 *  Created on: Aug 13, 2019
 *      Author: joseph
 */

#ifndef LAYER_LOADER_H_
#define LAYER_LOADER_H_

#include <string>
#include <ostream>
#include <sys/types.h>
#include <sys/stat.h>

#if __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
#define BC_USE_EXPERIMENTAL_FILE_SYSTEM 1
#endif

namespace BC {
namespace nn {

struct Layer_Loader {

	std::string root_directory;
	std::string current_layername;
	int current_layer_index = -1;

	Layer_Loader(std::string root_directory_):
		root_directory(root_directory_) {}

	void set_current_layer_name(std::string current_layername_){
		current_layername = current_layername_;
	}

	void set_current_layer_index(int layer_index){
		current_layer_index = layer_index;
	}

	std::string current_layer_subdir() const {
		std::string subdir = "l" + std::to_string(current_layer_index) + "_" + current_layername;
		return root_directory + bc_directory_separator() + subdir;
	}

	int make_current_directory(){
		std::string mkdir = "mkdir " + current_layer_subdir();
		return system(mkdir.c_str());
	}

	template<class T>
	void save_variable(const T& tensor, std::string variable_name) {
		std::ofstream output(path_from_args(tensor, variable_name));
		output << tensor.to_raw_string();
	}

	template<class T>
	void load_variable(T& tensor, std::string variable_name) {
		load_variable(tensor, variable_name, BC::traits::Integer<T::tensor_dimension>());
	}

	template<class T>
	void load_variable(T& tensor, std::string variable_name, BC::traits::Integer<1>) {
		using value_type = typename T::value_type;
		auto descriptor = BC::io::csv_descriptor(path_from_args(tensor, variable_name)).header(false);
		tensor = T(BC::io::read_uniform<value_type>(descriptor, tensor.get_allocator()).row(0));
	}

	template<class T>
	void load_variable(T& tensor, std::string variable_name, BC::traits::Integer<2>) {
		using value_type = typename T::value_type;
		auto descriptor = BC::io::csv_descriptor(path_from_args(tensor, variable_name)).header(false);
		tensor = BC::io::read_uniform<value_type>(descriptor, tensor.get_allocator());
	}


private:

	std::string path_from_args(int dimension, std::string variable_name) {
		return current_layer_subdir()
				+ bc_directory_separator()
				+ variable_name
				+ "." + dimension_to_tensor_name(dimension);
	}

	template<class T>
	std::string path_from_args(const T& tensor,std::string variable_name) {
		return path_from_args(T::tensor_dimension, variable_name);
	}

	bool file_exists(int dimension, std::string filename) {
		return file_exists(path_from_args(dimension, filename));
	}

	bool file_exists(std::string filename) {
#ifdef BC_USE_EXPERIMENTAL_FILE_SYSTEM
		return std::experimental::filesystem::exists(filename);
#else
		return std::ifstream(filename).good();
#endif
	}

	static std::string dimension_to_tensor_name(int dimension) {
		switch(dimension) {
		case 0: return "scl";
		case 1: return "vec";
		case 2: return "mat";
		case 3: return "cube";
		default: return "t" + std::to_string(dimension);
		}
	}
};


}
}



#undef BC_USE_EXPERIMENTAL_FILE_SYSTEM
#endif /* LAYER_LOADER_H_ */
