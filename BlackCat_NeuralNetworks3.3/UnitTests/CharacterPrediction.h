#include "../BlackCat_NeuralNetworks.h"
#include <fstream>
#include <iostream>
#include <string>
using BC::NN::vec;
using BC::NN::scal;
using BC::NN::mat;
typedef std::vector<vec> internal;
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

template<class T>
int test(T& words, int helper = 1) {

	const int ITERATIONS = 500;
	const int NUMB_WORDS = words.size();
	std::ifstream is("EANfin.bc");

	//Create a Neural Network
	NeuralNetwork<FeedForward, GRU, FeedForward> network(alpha_length, 128, 64, alpha_length);

	network.read(is);
	network.setLearningRate(.001);



	for (int i = 0; i < ITERATIONS; ++i) {
		std::cout << " i = " << i<< std::endl;

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

	//save the neural network
	std::ofstream os(std::string("EANfin.bc"));
	network.write(os);
	os.close();



	//-----------------------------------------------RESULTS-----------------------------------------------//
	//--->Would comment out the training during result tests<---//
	int helper_count = helper;				//Number of characters given
	float n_words = 0;						//Number of words
	float correct = 0;						//Number of correct words
	float char_correct = 0;					//Number of correct chars
	float chars = 0;						//number of chars
	float valid_words = 0;					//number of words predicted by NN that were in the corpus
	//print out/test the predictions
	int myj;

	for (int j = 0; j < NUMB_WORDS; ++j) {
		std::cout << words[j] << " | ";

		//give it the first n characters of a word
		vec input;
		std::string wrd = "";

		for (int x = 0; x < helper_count; ++x) {
			if (x >=  words[j].length())
				break;

			wrd += words[j][x];
			input = char_to_vec(words[j][x]);
			input = network.forwardPropagation_Express(input);
		}

		std::cout << wrd; //print current word (the fragment of the word we fed to the NN)

		//Recursively feed output to inputs
		if (helper_count < words[j].length()) {
			wrd += vec_to_char(input);

			for (int x = helper_count; x < words[j].length() - 1; ++x) {
			input = network.forwardPropagation_Express(input);
			wrd += vec_to_char(input);

			//uncomment this for truncated results
			//if (words[j].size() < 6)
				chars ++;

			//uncomment this for truncated results
			//if (words[j].size() < 6)
			if (words[j][x] == wrd[wrd.length() - 1]) {
				char_correct += 1;
			}
		}

			//uncomment this for truncated results
			//if (words[j].size() < 6)
			if (words[j].size() > helper_count ) {
				n_words++;

				if (wrd == words[j]) {
					correct += 1;
					valid_words++;
				} else
				for (auto& w: words) {
					if (wrd == w) {
						valid_words++;
						break;
					}
				}
			}
		}


		std::cout << std::endl;

	}
	std::cout << " helper count == " << helper << "----------------------------------" << std::endl;
	std::cout << " words correct " << correct / n_words << std::endl;
	std::cout << " chars correct " << char_correct / chars << std::endl;
	std::cout << " valid _wrods =" << valid_words << std::endl;
	std::cout << " valid words " << valid_words / n_words << std::endl;

	return 0;
}


}
}
}
