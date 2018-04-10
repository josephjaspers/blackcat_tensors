#include "stdafx.h"
#include "LSTM.h"
const Vector & LSTM::Xt()
{
	if (bpX.empty()) {
		return INPUT_ZEROS;
	}
	return bpX.back();
}

const Vector & LSTM::Ct()
{
	if (bpC.empty()) {
		return OUTPUT_ZEROS;
	}
	return bpC.back();
}

const Vector & LSTM::Ct_1()
{
	if (bpC.size() < 2) {
		return OUTPUT_ZEROS;
	}
	else {
		return bpC[bpC.size() - 2]; //return second to output 
	}
}

const Vector & LSTM::Ft()
{
	if (bpF.empty()) {
		return OUTPUT_ZEROS;
	}
	else {
		return bpF.back();
	}
}

const Vector & LSTM::Zt()
{
	if (bpZ.empty()) {
		return OUTPUT_ZEROS;
	}
	else {
		return bpZ.back();
	}
}

const Vector & LSTM::It()
{
	if (bpI.empty()) {
		return OUTPUT_ZEROS;
	}
	else {
		return bpI.back();
	}
}

const Vector & LSTM::Ot()
{
	if (bpO.empty()) {
		return OUTPUT_ZEROS;
	}
	else {
		return bpO.back();
	}
}

const Vector & LSTM::Yt()
{
	if (bpY.empty()) {
		return OUTPUT_ZEROS;
	}
	else {
		return bpY.back();
	}
}


LSTM::LSTM(int inputs, int outputs) : Layer(inputs, outputs)
{
	z_g.setTanh();
	i_g.setSigmoid();
	f_g.setSigmoid();
	o_g.setSigmoid();
	g.setTanh();

	bz_gradientStorage = Vector(outputs);
	wz_gradientStorage = Matrix(outputs, inputs);
	rz_gradientStorage = Matrix(outputs, outputs);

	bi_gradientStorage = Vector(outputs);
	wi_gradientStorage = Matrix(outputs, inputs);
	ri_gradientStorage = Matrix(outputs, outputs);

	bf_gradientStorage = Vector(outputs);
	wf_gradientStorage = Matrix(outputs, inputs);
	rf_gradientStorage = Matrix(outputs, outputs);

	bo_gradientStorage = Vector(outputs);
	wo_gradientStorage = Matrix(outputs, inputs);
	ro_gradientStorage = Matrix(outputs, outputs);

	c = Vector(outputs);
	dc = Vector(outputs);

	x = Vector(inputs);
	y = Vector(outputs);

	z = Vector(outputs);
	bz = Vector(outputs);
	dz = Vector(outputs);
	wz = Matrix(outputs, inputs);
	rz = Matrix(outputs, outputs);

	i = Vector(outputs);
	bi = Vector(outputs);
	di = Vector(outputs);
	wi = Matrix(outputs, inputs);
	ri = Matrix(outputs, outputs);

	f = Vector(outputs);
	bf = Vector(outputs);
	df = Vector(outputs);
	wf = Matrix(outputs, inputs);
	rf = Matrix(outputs, outputs);

	o = Vector(outputs);
	bo = Vector(outputs);
	od = Vector(outputs);
	wo = Matrix(outputs, inputs);
	ro = Matrix(outputs, outputs);

	Matrices::randomize(bz, 0, 4);
	Matrices::randomize(wz, 0, 4);
	Matrices::randomize(rz, 0, 4);

	Matrices::randomize(bi, -4, 4);
	Matrices::randomize(wi, -4, 4);
	Matrices::randomize(ri, -4, 4);

	//initialize forget gate in negative range so network must be trained to "remember"
	Matrices::randomize(bf, -4, 0);
	Matrices::randomize(wf, -4, 0);
	Matrices::randomize(rf, -4, 0);
	//start in positive range (output everything)
	Matrices::randomize(bo, 0, 5);
	Matrices::randomize(wo, 0, 5);
	Matrices::randomize(ro, 0, 5);
}

Vector LSTM::forwardPropagation_express(const Vector & x)
{
	f_g(f = (wf * x + rf * y + bf));
	z_g(z = (wz * x + rz * y + bz));
	i_g(i = (wi * x + ri * y + bi));
	o_g(o = (wo * x + ro * y + bo));

	c &= f;
	c += (z & i);

	y = (g.nonLin(c) & o); //apply non linearity to a copy of the cell state (g(Vector& x) -- accepts a reference to x while g.nonLin() returns a crushed copy

	if (next != nullptr)
		return next->forwardPropagation_express(y);
	else
		return y;
}

Vector LSTM::forwardPropagation(const Vector & input)
{
	updateBPStorage(); //stores all the current activations 

	x = input;
	f_g(f = (wf * x + rf * y + bf));
	z_g(z = (wz * x + rz * y + bz));
	i_g(i = (wi * x + ri * y + bi));
	o_g(o = (wo * x + ro * y + bo));

	c &= f;
	c += (z & i);

	y = g.nonLin(c) & o; //g -parenthesis operator applies the nonlinearity function to a reference to the parameter, nonLin creates and returns a copy. (This preserves the original cell state)

	//continue forwardprop
	if (next != nullptr)
		return next->forwardPropagation(y);
	else
		return y;
}

Vector LSTM::backwardPropagation(const Vector & dy)
{
	//calculate gate errors
	dc = dy & o & g.d(g.nonLin(c));
	od = dy & g.nonLin(c) & o_g.d(o);
	df = dc & Ct() & f_g.d(f);
	dz = dc & i & z_g.d(z);
	di = dc & z & i_g.d(i);
	dc &= f; //update cell error
	//Store gradients
	storeGradients();
	//calculate input error
	Vector dx = (wz ->* dz + wf ->* df + wi ->* di + wo ->* od);


	//continue backpropagation
	if (prev != nullptr) {
		return prev->backwardPropagation(dx);
	}
	else
		return dx;
}

Vector LSTM::backwardPropagation_ThroughTime(const Vector & deltaError)
{
	//calculate delta 
	Vector dy = deltaError + rz ->* dz + ri ->* di + rf ->* df + ro ->* od;
	//math of error 
	dc += dy & g.d(y) & Ot() & g.d(g.nonLin(Ct()));
	od = dc & g.nonLin(Ct()) & o_g.d(Ot());
	df = dc & Ct_1() & f_g.d(Ft());
	dz = dc & It() & z_g.d(Zt());
	di = dc & Zt() & i_g.d(It());
	//Store gradients 

	//get input error
	Vector dx = (wz ->* dz) + (wf ->* df) + (wi ->* di) + (wo ->* od);
	//send the error through the gate 
	dc &= Ft();
	//update backprop storage
	bpStorage_pop_back_all();
	//continue backpropagation
	if (prev != nullptr) {
		return prev->backwardPropagation_ThroughTime(dx);
	}
	else
		return dx;
}

void LSTM::clearBPStorage()
{
	bpF.clear();
	bpZ.clear();
	bpX.clear();
	bpC.clear();
	bpI.clear();
	bpO.clear();

	Layer::clearBPStorage();
}

void LSTM::clearGradients()
{
	Matrix::fill(wz_gradientStorage, 0);
	Matrix::fill(rz_gradientStorage, 0);
	Vector::fill(bz_gradientStorage, 0);

	Matrix::fill(wi_gradientStorage, 0);
	Matrix::fill(ri_gradientStorage, 0);
	Vector::fill(bi_gradientStorage, 0);

	Matrix::fill(wf_gradientStorage, 0);
	Matrix::fill(rf_gradientStorage, 0);
	Vector::fill(bf_gradientStorage, 0);

	Layer::clearGradients();
}

void LSTM::updateGradients()
{
	wz += wz_gradientStorage & lr;
	bz += bz_gradientStorage & lr;
	rz += rz_gradientStorage & lr;

	wi += wi_gradientStorage & lr;
	bi += bi_gradientStorage & lr;
	ri += ri_gradientStorage & lr;

	wf += wf_gradientStorage & lr;
	bf += bf_gradientStorage & lr;
	rf += rf_gradientStorage & lr;

	Layer::updateGradients();
}

LSTM* LSTM::read(std::ifstream & is)
{
	int inputs, outputs;
	is >> inputs;
	is >> outputs;

	LSTM* lstm = new LSTM(inputs, outputs);

	lstm->c = Vector::read(is);
	lstm->x = Vector::read(is);
	lstm->y = Vector::read(is);
	lstm->f = Vector::read(is);
	lstm->i = Vector::read(is);
	lstm->z = Vector::read(is);
	lstm->o = Vector::read(is);

	lstm->bz = Vector::read(is); lstm->bi = Vector::read(is); lstm->bf = Vector::read(is); lstm->bo = Vector::read(is);
	lstm->wz = Matrix::read(is); lstm->wi = Matrix::read(is); lstm->wf = Matrix::read(is); lstm->wo = Matrix::read(is);
	lstm->rz = Matrix::read(is); lstm->ri = Matrix::read(is); lstm->rf = Matrix::read(is); lstm->ro = Matrix::read(is);

	lstm->g.read(is);
	lstm->z_g.read(is);
	lstm->i_g.read(is);
	lstm->f_g.read(is);
	lstm->o_g.read(is);

	return lstm;
}

void LSTM::write(std::ofstream & os)
{
	os << NUMB_INPUTS << ' ';
	os << NUMB_OUTPUTS << ' ';

	c.write(os);
	x.write(os);
	y.write(os);
	f.write(os);
	i.write(os);
	z.write(os);
	o.write(os);

	bz.write(os); bi.write(os); bf.write(os); bo.write(os);
	wz.write(os); wi.write(os); wf.write(os); wo.write(os);
	rz.write(os); ri.write(os); rf.write(os); ro.write(os);

	g.write(os);
	z_g.write(os);
	i_g.write(os);
	f_g.write(os);
	o_g.write(os);

}

void LSTM::writeClass(std::ofstream & os)
{
	os << 2 << ' ';
}

void LSTM::set_ForgetGate_Sigmoid()
{
	f_g.setSigmoid();
}

void LSTM::set_ForgetGate_Tanh()
{
	f_g.setTanh();
}

void LSTM::set_InputGate_Sigmoid()
{
	i_g.setSigmoid();
}

void LSTM::set_InputGate_Tanh()
{
	i_g.setTanh();
}

void LSTM::set_OutputGate_Sigmoid()
{
	o_g.setSigmoid();
}

void LSTM::set_OutputGate_Tanh()
{
	o_g.setTanh();
}

void LSTM::storeGradients()
{
	wz_gradientStorage -= dz * x;
	bz_gradientStorage -= dz;
	rz_gradientStorage -= dz * c;

	wf_gradientStorage -= df * x;
	bf_gradientStorage -= df;
	rf_gradientStorage -= df * c;

	wi_gradientStorage -= di * x;
	bi_gradientStorage -= di;
	ri_gradientStorage -= di * c;

	wo_gradientStorage -= od * x;
	bo_gradientStorage -= od;
	ro_gradientStorage -= od * c;
}

void LSTM::storeGradients_BPTT()
{
	wz_gradientStorage -= dz * Xt();
	bz_gradientStorage -= dz;
	rz_gradientStorage -= dz * Yt();

	wi_gradientStorage -= di * Xt();
	bi_gradientStorage -= di;
	ri_gradientStorage -= di * Yt();

	wi_gradientStorage -= od * Xt();
	bi_gradientStorage -= od;
	ri_gradientStorage -= od * Yt();

	wf_gradientStorage -= df * Xt();
	bf_gradientStorage -= df;
	rf_gradientStorage -= df * Yt();
}

void LSTM::bpStorage_pop_back_all()
{
	bpX.pop_back();
	bpC.pop_back();
	bpF.pop_back();
	bpZ.pop_back();
	bpI.pop_back();
	bpY.pop_back();
}

void LSTM::updateBPStorage()
{
	bpX.push_back(x);
	bpC.push_back(c);
	bpF.push_back(f);
	bpZ.push_back(z);
	bpI.push_back(i);
	bpO.push_back(o);
	bpY.push_back(y);
}

