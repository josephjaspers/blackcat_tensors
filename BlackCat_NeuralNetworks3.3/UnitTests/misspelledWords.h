#include "../BlackCat_NeuralNetworks.h"
#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>
using BC::NN::vec;
using BC::NN::scal;
using BC::NN::mat;
typedef std::vector<vec> data;
typedef vec tensor;

namespace BC {
namespace NN {
namespace Word_Test {

std::string alphabet = { "abcdefghijklmnopqrstuvwxyz" };

vec char_to_vec(char value) {
	vec out(26); out.zero();

	for (int i = 0; i < 26; ++i)
		if (value == alphabet[i])
			out(i) = 1;

	return out;
}
char vec_to_char(const vec& v) {
	float max = 0;
	int index = - 1;

	for (int i = 0; i < 26; ++i)
		if (max < v(i)) {
			max = v(i);
			index = i;
		}

	std::cout << alphabet[index];
	return alphabet[index];
}




int test() {

	//Here are some words
	std::string words[7] = { "wort", "tore", "sort", "hurt", "lore", "store", "galore" };

	//Create a Neural Network
	NeuralNetwork<FeedForward, GRU, FeedForward> network(26, 30, 30, 26);
	network.setLearningRate(.03);


	//Train 4k iterations
	for (int i = 0; i < 4000; ++i) {
		//7 for the number of words in corpus
		for (int j = 0; j < 7; ++j) {
			for (int x = 0; x < words[j].length() - 1; ++x) {
				vec input = char_to_vec(words[j][x]);	//convert current char to 1-hot
				network.forwardPropagation(input);		//propagate
			}
			for (int x = words[j].length()-1; x > 0; --x){
				vec output = char_to_vec(words[j][x]);	//back propagate backwards thru word
				network.backPropagation(output);
			}

			network.updateWeights();
			network.clearBPStorage();
		}
	}


	for (int j = 0; j < 7; ++j) {
		std::cout << words[j] << " | ";
		std::cout << words[j][0];
				for (int x = 0; x < words[j].length() - 1; ++x) {
					auto out = network.forwardPropagation_Express(char_to_vec(words[j][x]));
					 vec_to_char(out);
				}

				std::cout << std::endl;
				if (j % 100 == 0) {
				network.updateWeights();
				network.clearBPStorage();
				}
			}
		}

}
}
}
