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

//convert a char to a 1-hot vector (index = the index in the alphbet)
vec char_to_vec(char value) {

	vec out(27);
	out.zero();

	for (int i = 0; i < 27; ++i)
		if (value == alphabet[i])
			out(i) = 1;

	return out;
}

//convert a 1-hot vector to a character (for the output)
char vec_to_char(const vec& v) {
	float max = 0;
	int index = -1;

	for (int i = 0; i < 27; ++i)
		if (max < v(i)) {
			max = v(i);
			index = i;
		}

	std::cout << alphabet[index];
	return alphabet[index];
}

int test() {

	const int ITERATIONS = 1000;

	std::vector<std::string> words = { "have", "a", "new", "fear", "when", "it", "comes", "to", "their", "care", "the", "obama", "they", "might", "win", "the", "trump", "could",
			"to", "no", "the", "the", "suit", "which", "the", "to", "spend", "of", "on", "for", "and", "house", "a", "big", "on", "but", "a", "loss", "of", "the", "could", "cause",
			"the", "care", "to", "of", "to", "have", "a", "that", "could", "lead", "to", "chaos", "in", "the", "and", "spur", "a", "just", "as", "gain", "full", "of", "the", "to",
			"stave", "off", "that", "could", "find", "in", "the", "of", "huge", "sums", "to", "prop", "up", "the", "obama", "care", "law", "who", "have", "been", "an", "end", "to",
			"the", "law", "for", "years", "in", "twist", "j", "about", "could", "to", "fight", "its", "in", "the", "house", "on", "some", "in", "the", "eager", "to", "avoid", "an",

						"ugly", "on", "hill", "and", "the", "trump", "team", "are", "out", "how", "to", "the", "which", "after", "the", "has", "been", "put", "in", "limbo", "until", "at",
			"least", "late", "by", "the", "court", "of", "for", "the", "of", "they", "are", "not", "yet", "ready", "to", "their", "given", "that", "this", "the", "obama", "and",
			"it", "would", "be", "to", "said", "j", "a", "for", "the", "trump", "upon", "the", "trump", "will", "this", "case", "and", "all", "of", "the", "care", "act", "in", "a",
			"in", "judge", "m", "ruled", "that", "house", "had", "the", "to", "sue", "the", "over", "a", "and", "that", "the", "obama", "had", "been", "the", "in", "of", "the",
			"from", "the", "that", "judge", "would", "be", "and", "the", "have", "in", "place", "the", "in", "a", "halt", "in", "the", "after", "mr", "trump", "won", "house",
			"last", "month", "told", "the", "court", "that", "they", "and", "the", "s", "team", "are", "for", "of", "this", "to", "take", "after", "the", "s", "on", "jan", "the",
			"of", "the", "case", "house", "said", "will", "the", "and", "his", "time", "to", "to", "or", "to", "this", "in", "the", "house", "the", "of", "if", "the", "which",
			"have", "an", "are", "that", "the", "in", "for", };

	const int NUMB_WORDS = words.size();

	//Create a Neural Network
	NeuralNetwork<FeedForward, GRU, FeedForward> network(27, 240, 120, 27);
	network.setLearningRate(.003);
	network.set_omp_threads(8);
	omp_set_num_threads(8);
	//Train 4k iterations
	const int batch = 16;
	for (int i = 0; i < ITERATIONS; ++i) {
//		std::cout << " iteartion = " << i << std::endl;
		for (int j = 0; j < NUMB_WORDS - batch; j += batch) {
#pragma omp parallel for
			for (int k = j; k < j + batch; ++k) {
			for (int x = 0; x < words[j].length() - 1; ++x) {
				vec input = char_to_vec(words[j][x]);	//convert current char to 1-hot
				network.forwardPropagation(input);		//propagate

			}
			vec input = char_to_vec(' ');	//convert current char to 1-hot
			network.forwardPropagation(input);		//propagate
			network.backPropagation(input);

			//iterate backwards through the word
			for (int x = words[j].length() - 1; x > 0; --x) {
				vec output = char_to_vec(words[j][x]);	//back propagate backwards thru word
				network.backPropagation(output);
			}
			}
#pragma omp barrier
			//update and stuff
			network.updateWeights();
			network.clearBPStorage();
		}
	}

	//print out/test the predictions
	for (int j = 0; j < NUMB_WORDS; ++j) {
		std::cout << "{\'" << words[j] << "\', \'";
		std::cout << words[j][0];
		for (int x = 0; x < words[j].length() - 1; ++x) {
			auto out = network.forwardPropagation_Express(char_to_vec(words[j][x]));
			vec_to_char(out);
		}
		std::cout << "\' }," << std::endl;
	}
	return 0;
}

}
}
}
