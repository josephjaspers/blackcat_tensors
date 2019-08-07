import blackcat_tensors
from blackcat_tensors import bc #bc is the c++ namespace 
import pandas as pd
import numpy as np

df = pd.read_csv('../datasets/heart_disease.csv')

"""
Source: https://www.kaggle.com/ronitf/heart-disease-uci

age age in years
sex(1 = male; 0 = female)
cp chest pain type
trestbps resting blood pressure (in mm Hg on admission to the hospital)
cholserum cholestoral in mg/dl
fbs (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg resting electrocardiographic results
thalach maximum heart rate achieved
exang exercise induced angina (1 = yes; 0 = no)
old peakST depression induced by exercise relative to rest
slope the slope of the peak exercise ST segment
can umber of major vessels (0-3) colored by flourosopy
thal 3 = normal; 6 = fixed defect; 7 = reversable defect
target 1 or 0
"""


def normalize(col):
	# Squashes range of all data inbetwee 0-1 
	max_ = col.max()
	min_ = col.min()
	return  (col - min_) / (max_ - min_)

def normalize_columns(df):
	for c in df.keys():
		df[c] = normalize(df[c]) 
	return df 

df = pd.read_csv('../datasets/heart_disease.csv') 
df = normalize_columns(df)

numb_rows = len(df)
numb_cols = len(df.keys())

# Note: rows/cols are swapped (we are transposing the df)
# each column of the inputs-Matrix should be a single patient 
inputs = blackcat_tensors.Matrix(numb_cols-1, numb_rows) # minus one to exclude the 'target' column
outputs = blackcat_tensors.from_np(df.target.values.astype(np.float64)) 
del df['target'] # remove the output column

# swap cols/rows as we are transposing the Matrix, 
df = pd.DataFrame(df.T)
for col, k in zip(inputs, df.keys()):
	col.assign(blackcat_tensors.from_np(df[k].values.astype(np.float64)))

#debug output
print(df)
print(inputs[0])

network = bc.nn.neuralnetwork(
	bc.nn.feedforward(13, 32),
	bc.nn.tanh(32),
	bc.nn.feedforward(32, 1),
	bc.nn.logistic(1),
	bc.nn.outputlayer(1)
)

network.set_batch_size(1) 

epochs = 3 
samples = numb_rows 
from random import randrange 
for epoch in range(epochs):
	for i in range(samples):
		mat = blackcat_tensors.Matrix(13, 1)
		mat[0] = inputs[i]
		network.forward_propagation(mat)
	
		o = blackcat_tensors.Matrix(1,1)
		o[0] = outputs[i].memptr()[0] 
		network.back_propagation(o[0])
		#network.forward_propagation(inputs[i])
		#network.back_propagation(outputs[i:i+1]) # use a 'ranged-slice' so a Vector is returned instead of a Scalar 
		network.update_weights() 


#todo split inputs into train/test 
#create better test statistics 
#add apropriate training/error methods 
tests=150
for t in range(tests):
	hyp = network.forward_propagation(inputs[i])
	
	print('Sample: ', t) 
	hyp.print()
	outputs[i].print() 







