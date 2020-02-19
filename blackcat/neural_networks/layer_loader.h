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

namespace fs = bc::filesystem;

struct Layer_Loader {

	string root_directory;
	string current_layername;
	int current_layer_index = -1;

	Layer_Loader(string root_directory):
		root_directory(root_directory) {}

	void set_current_layer_name(string current_layername) {
		this->current_layername = current_layername;
	}

	void set_current_layer_index(int layer_index) {
		this->current_layer_index = layer_index;
	}

	string current_layer_subdir() const 
	{
		string index = std::to_string(current_layer_index);
		string subdir = "l" + index  + "_" + current_layername;
		return fs::make_path(root_directory, subdir);
	}

	template<class T>
	void save_variable(const T& tensor, string variable_name) {
		std::ofstream output(path_from_args(tensor, variable_name));
		output << tensor.to_raw_string();
	}

	template<class T>
	void load_variable(T& tensor, string variable_name) {
		load_variable(tensor, variable_name, Integer<T::tensor_dim>());
	}

	template<class Tensor>
	void load_variable(Tensor& tensor, string variable_name, Integer<1>) 
	{
		using value_type = typename Tensor::value_type;
		string filename = path_from_args(tensor, variable_name);
		auto descriptor = bc::io::csv_descriptor(filename).header(false);
		tensor = Tensor(bc::io::read_uniform<value_type>(descriptor, tensor.get_allocator()).row(0));
	}

	template<class T>
	void load_variable(T& tensor, string variable_name, Integer<2>) {
		using value_type = typename T::value_type;
		auto descriptor = bc::io::csv_descriptor(path_from_args(tensor, variable_name)).header(false);
		tensor = bc::io::read_uniform<value_type>(descriptor, tensor.get_allocator());
	}

	template<class T, int X>
	void load_variable(T& tensor, string variable_name, Integer<X>) {
		using value_type = typename T::value_type;
		auto descriptor = bc::io::csv_descriptor(path_from_args(tensor, variable_name)).header(false);
		auto csv_mat = bc::io::read_uniform<value_type>(descriptor, tensor.get_allocator());
		tensor = csv_mat.reshaped(tensor.inner_shape());
	}

	void make_current_directory() 
	{
		if (!fs::directory_exists(current_layer_subdir()))
			fs::mkdir(current_layer_subdir());
	}

private:

	string path_from_args(int dim, string variable_name) 
	{
		string extension = dim_to_extension(dim);
		string directory = current_layer_subdir();
		return fs::make_path(directory, variable_name + "." + extension);
	}

	template<class T>
	string path_from_args(const T& tensor, string variable_name) {
		return path_from_args(T::tensor_dim, variable_name);
	}

public:

	bool file_exists(int dim, string filename) {
		return fs::file_exists(path_from_args(dim, filename));
	}


	static string dim_to_extension(int dim) {
		switch(dim) {
			case 0: return "scl";
			case 1: return "vec";
			case 2: return "mat";
			case 3: return "cube";
			default: return "t" + std::to_string(dim);
		}
	}
};


}
}



#undef BC_USE_EXPERIMENTAL_FILE_SYSTEM
#endif /* LAYER_LOADER_H_ */
