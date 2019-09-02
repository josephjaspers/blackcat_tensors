import blackcat_tensors
from blackcat_tensors import bc #bc is the c++ namespace 
import pandas as pd
import numpy as np

df = pd.read_csv('../examples/mnist_test/mnist_train.csv')

outputs = blackcat_tensors.Matrix(10, len(df))
inputs  = blackcat_tensors.Matrix(784, len(df))
single_Value_outputs = df.label.values 

outputs.zero()
#'intializing one hot vector of outputs'
for i in range(0, len(df.label)):
	outputs[i][int(df.label.values[i])].fill(1)

#copy each image (column) into the 'input matrix'
del df['label']
df = pd.DataFrame(df.T) 
for i in range(inputs.cols()):
	inputs[i] = blackcat_tensors.from_np(df[i].values.astype(np.float64))

# normalize the data 
inputs /= 255 

network = bc.nn.neuralnetwork(
	bc.nn.feedforward(784, 256),
	bc.nn.logistic(256),
	bc.nn.feedforward(256, 10),
	bc.nn.softmax(10),
	bc.nn.output_layer(10)
)

samples = outputs.cols()
batch_size = 8
epochs = 10 
batches = int(samples/batch_size)
network.set_batch_size(batch_size) 

outs = bc.tensors.reshape(outputs, bc.tensors.shape(10, batch_size, batches))
ins  = bc.tensors.reshape(inputs, bc.tensors.shape(784, batch_size, batches))


print('training...')
for i in range(epochs):
	print('epoch:', i)
	for batch in range(batches): 
		network.forward_propagation(ins[batch])
		network.back_propagation(outs[batch])
		network.update_weights();


prediction = network.forward_propagation(ins[0])
prediction.print()
for i in range(batch_size):	
	print('----------------------')
	print('EXPECTED ', outs[0][i])
	print('PREDICTED', prediction[i])

