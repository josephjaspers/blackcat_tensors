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
std::string alphabet = { "abcdefghijklmnopqrstuvwxyz " };
int alpha_length = alphabet.length();

//convert a char to a 1-hot vector (index = the index in the alphbet)
vec char_to_vec(char value) {

	vec out(alpha_length);
	out.zero();

	for (int i = 0; i < alpha_length; ++i)
		if (value == alphabet[i])
			out(i) = 1;

	return out;
}

//convert a 1-hot vector to a character (for the output)
char vec_to_char(const vec& v) {
	float max = 0;
	int index = - 1;

	for (int i = 0; i < alpha_length; ++i)
		if (max < v(i)) {
			max = v(i);
			index = i;
		}

	std::cout << alphabet[index];
	return alphabet[index];
}



#include "Data/fixed_TheRaven.h"
#include "Data/fixed_BlackCat.h"
#include "Data/fixed_ThePit.h"

template<class T>
int test(T& words) {

	const int ITERATIONS = 100;
	const int NUMB_WORDS = words.size();


	//Create a Neural Network
	NeuralNetwork<FeedForward, GRU, FeedForward> network(alpha_length, 128, 64, alpha_length);

	std::ifstream is("EdgarAllenNetworkfin.bc");
	network.read(is);



	network.setLearningRate(.001);
	//Train 4k iterations
	for (int i = 0; i < ITERATIONS; ++i) {
		std::cout << " i = " << i<< std::endl;
//		if (i% 10 == 0) {
//			std::cout << " iteration = " << i << std::endl;
//			std::ofstream os("EdgarAllenNetwork" + std::to_string(i) + ".bc");
//			network.write(os);
//			os.close();
//		}
		for (int j = 0; j < NUMB_WORDS; ++j) {
//			std::cout << " curr word = " << j << " outof " << NUMB_WORDS << std::endl;
			for (int x = 0; x < words[j].length(); ++x) {
				vec input = char_to_vec(words[j][x]);	//convert current char to 1-hot
				network.forwardPropagation(input);		//propagate
			}
			vec input (char_to_vec(' '));
			network.backPropagation(input);
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


	std::ofstream os(std::string("EdgarAllenNetwork") + std::string("fin") + std::string(".bc"));
	network.write(os);
	os.close();


	int helper_count = 1;
	//print out/test the predictions
	for (int j = 0; j < NUMB_WORDS; ++j) {
		std::string wrd = "";
		std::cout << words[j] << " | ";
//		std::cout << words[j][0];

		//give it the first three characters of a word
		vec input;
		for (int x = 0; x < helper_count; ++x) {
			if (x >=  words[j].length())
				break;

			wrd += words[j][x];
			input = char_to_vec(words[j][x]);
			input = network.forwardPropagation_Express(input);
			vec_to_char(input);
		}
		//let it predict the rest
		if (helper_count < words[j].length()) {
			std::cout << words[j][helper_count];
		for (int x = helper_count; x < words[j].length() - 1; ++x) {
			input = network.forwardPropagation_Express(input);
			if (vec_to_char(input) == ' ') {
//				if (x == words[j].length() - 2) {
					std::cout << " || ";
//				}
				break;
			}
		}
		}
		std::cout << std::endl;

	}


	return 0;
}


}
}
}
