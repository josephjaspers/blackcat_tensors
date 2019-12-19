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

namespace bc {
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
		return bc::filesystem::make_path(root_directory, subdir);
	}

	template<class T>
	void save_variable(const T& tensor, std::string variable_name) {
		std::ofstream output(path_from_args(tensor, variable_name));
		output << tensor.to_raw_string();
	}

	template<class T>
	void load_variable(T& tensor, std::string variable_name) {
		load_variable(tensor, variable_name, bc::traits::Integer<T::tensor_dimension>());
	}

	template<class T>
	void load_variable(T& tensor, std::string variable_name, bc::traits::Integer<1>) {
		using value_type = typename T::value_type;
		auto descriptor = bc::io::csv_descriptor(path_from_args(tensor, variable_name)).header(false);
		tensor = T(bc::io::read_uniform<value_type>(descriptor, tensor.get_allocator()).row(0));
	}

	template<class T>
	void load_variable(T& tensor, std::string variable_name, bc::traits::Integer<2>) {
		using value_type = typename T::value_type;
		auto descriptor = bc::io::csv_descriptor(path_from_args(tensor, variable_name)).header(false);
		tensor = bc::io::read_uniform<value_type>(descriptor, tensor.get_allocator());
	}

	void make_current_directory() {
		if (!bc::filesystem::directory_exists(current_layer_subdir()))
			bc::filesystem::mkdir(current_layer_subdir());
	}

private:

	std::string path_from_args(int dimension, std::string variable_name) {
		std::string extension = dimension_to_tensor_name(dimension);
		return bc::filesystem::make_path(
				current_layer_subdir(), variable_name + "." + extension);
	}

	template<class T>
	std::string path_from_args(const T& tensor,std::string variable_name) {
		return path_from_args(T::tensor_dimension, variable_name);
	}

public:

	bool file_exists(int dimension, std::string filename) {
		return bc::filesystem::file_exists(path_from_args(dimension, filename));
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
