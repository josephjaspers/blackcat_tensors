#include "../BlackCat_NeuralNetworks.h"
#include <fstream>
#include <iostream>
#include <string>
using BC::NN::vec;
using BC::NN::scal;
using BC::NN::mat;
typedef std::vector<vec> data;
typedef vec tensor;

namespace BC {
namespace NN {
namespace Word_Test {
std::string alphabet = { "abcdefghijklmnopqrstuvwxyz" };

//convert a char to a 1-hot vector (index = the index in the alphbet)
vec char_to_vec(char value) {

	vec out(26);
	out.zero();

	for (int i = 0; i < 26; ++i)
		if (value == alphabet[i])
			out(i) = 1;

	return out;
}

//convert a 1-hot vector to a character (for the output)
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

	const int ITERATIONS = 3000;
	const int NUMB_WORDS = 8;

	//Here are some words
	std::string words[NUMB_WORDS] = { "wartorn", "wort", "thecork", "fork", "baatorp", "asdgfd", "money", "galore" };

	//Create a Neural Network
	NeuralNetwork<FeedForward, GRU, FeedForward> network(26, 30, 30, 26);
	network.setLearningRate(.03);
	//Train 4k iterations
	for (int i = 0; i < ITERATIONS; ++i) {
		for (int j = 0; j < NUMB_WORDS; ++j) {
			for (int x = 0; x < words[j].length() - 1; ++x) {
				vec input = char_to_vec(words[j][x]);	//convert current char to 1-hot
				network.forwardPropagation(input);		//propagate
			}
			//iterate backwards through the word
			for (int x = words[j].length()-1; x > 0; --x){
				vec output = char_to_vec(words[j][x]);	//back propagate backwards thru word
				network.backPropagation(output);
			}
			//update and stuff
			network.updateWeights();
			network.clearBPStorage();
		}
	}



	//print out/test the predictions
	for (int j = 0; j < NUMB_WORDS; ++j) {
		std::cout << words[j] << " | ";
		std::cout << words[j][0];
		for (int x = 0; x < words[j].length() - 1; ++x) {
			auto out = network.forwardPropagation_Express(char_to_vec(words[j][x]));
			vec_to_char(out);
		}
		std::cout << std::endl;
	}
	return 0;
}

}
}
}
