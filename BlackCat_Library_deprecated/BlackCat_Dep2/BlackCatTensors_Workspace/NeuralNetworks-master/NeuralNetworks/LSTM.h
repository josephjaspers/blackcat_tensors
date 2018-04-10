#ifndef LSTMLayer_h
#define LSTMLayer_h
#include "Layer.h"

class LSTM : public Layer {

	nonLinearityFunct z_g; //input gate nonlinearity function 
	nonLinearityFunct i_g; //input gate nonlinearity function 
	nonLinearityFunct f_g; //forget nonlinearity funct 
	nonLinearityFunct o_g; //output gate nonLinearity function

	Vector c;				//cellstate
	Vector dc;				//error of cell state

	Vector y;				//current output (or output during BP)

	Vector x;				//current input
	Vector f, i, z, o;		//current activations for forget, input, write, output gate

	Vector bz, bi, bf, bo;	//Bias Vectors for gate
	Vector dz, di, df, od;	//error of each gate
	Matrix wz, wi, wf, wo;	//feed forward weights of each gate
	Matrix rz, ri, rf, ro;	//recurrent weights of each gate

	Vector bz_gradientStorage, bi_gradientStorage, bf_gradientStorage, bo_gradientStorage; //Bias gradient storage of gates
	Matrix wz_gradientStorage, wi_gradientStorage, wf_gradientStorage, wo_gradientStorage; //Weight gradient storage of gates
	Matrix rz_gradientStorage, ri_gradientStorage, rf_gradientStorage, ro_gradientStorage; //recurrnet gradient storage of gates

	//back propagation storages for...
	bpStorage bpX;	//inputs
	bpStorage bpC;	//cellstate
	bpStorage bpF;	//forget gate output
	bpStorage bpZ;	//write gate output
	bpStorage bpI;	//input gate output
	bpStorage bpO;	//output gate output
	bpStorage bpY;	//LSTM layer output


	const Vector& Xt(); //inputs at T
	const Vector& Ct();	//cellstate at T
	const Vector& Ct_1();//cellstate at T minus 1
	const Vector& Ft();	//forget activations at T
	const Vector& Zt();	//write activations at T
	const Vector& It();	//input activations at T
	const Vector& Ot();	//output activations at T
	const Vector& Yt();	//y activations at T

public:
	LSTM(int inputs, int outputs);
	Vector forwardPropagation_express(const Vector& x);
	Vector forwardPropagation(const Vector& x);
	Vector backwardPropagation(const Vector& dy);
	Vector backwardPropagation_ThroughTime(const Vector& dy);

	void clearBPStorage();
	void clearGradients();
	void updateGradients();
	
	static LSTM* read(std::ifstream& is);
	void write(std::ofstream& os);
	void writeClass(std::ofstream& os);

	void set_ForgetGate_Sigmoid();
	void set_ForgetGate_Tanh();

	void set_InputGate_Sigmoid();
	void set_InputGate_Tanh();
	
	void set_OutputGate_Sigmoid();
	void set_OutputGate_Tanh();

private:
	void storeGradients();			//encapsulation of storing gradients 
	void storeGradients_BPTT();		//encapsulation of storing gradients at BPTT
	void bpStorage_pop_back_all();	//encapsulation of removing the bp storages
	void updateBPStorage();			//encapsulation of updating the bpstorages
};
#endif


