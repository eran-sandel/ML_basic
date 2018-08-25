#!/usr/bin/python

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz

from IPython.display import Image, display
import pydotplus

import argparse

global args

def add_args(parser):
	parser.add_argument('--input', '-i', action="store",  required=True, help='input csv')
	parser.add_argument('--desc', action='store_true', default=False,dest='description',help='print description')
	parser.add_argument('--features','-f', action='append', required=True, dest='features',default=[],help='Add feature values')
	parser.add_argument('--target','-t', action='store', required=True, dest='target',help='Add target values')
	parser.add_argument('--mlnmin', action='store', default=20, required=False, dest='mlnmin',help='define min tree depth')
	parser.add_argument('--mlnmax', action='store', default=100, required=False, dest='mlnmax',help='define max tree depth')
	parser.add_argument('--model', action='store', default='DecisionTreeRegressor', required=False, dest='modeltype',help='DecisionTreeRegressor/DecisionTreeClassifier/RandomForestRegressor')

def set_model(mln,train_X,train_y,modeltype):
	# Specify Model, set max nodes
	#model = DecisionTreeRegressor(max_leaf_nodes=mln,random_state=1)
	if ( modeltype.lower() == "DecisionTreeRegressor".lower()):
		model = DecisionTreeRegressor(max_leaf_nodes=mln,random_state=1)
	elif ( modeltype.lower() == "DecisionTreeClassifier".lower()):
		model = DecisionTreeClassifier(max_leaf_nodes=mln,random_state=1)
	elif (modeltype.lower() == "RandomForestRegressor".lower()):
		model = RandomForestRegressor(max_leaf_nodes=mln,random_state=1)
	else:
		print("Invalid model, please select between DecisionTreeRegressor/DecisionTreeClassifier/RandomForestRegressor")
	# Fit Model
	model.fit(train_X, train_y)
	# Make validation predictions and calculate mean absolute error
	return model

def predict_model(model, val_X):
	val_predictions = model.predict(val_X)
	return val_predictions

def get_mae(val_predictions, val_y):
	val_mae = mean_absolute_error(val_predictions, val_y)
	return val_mae

def get_data_description(data):
	print(data.describe())
	print data.columns
	print(data.head())

def draw_tree(model,train_X, train_y):
	dot_data = StringIO()
	model.fit(train_X, train_y)
	dot_data = export_graphviz(model, out_file=None, 
                filled=True, rounded=True,
                special_characters=True)
	graph = pydotplus.graph_from_dot_data(dot_data)  
	Image(graph.create_png())


def open_file(file_path):
	data = pd.read_csv(file_path)
	#drop rows with N/A values
	data.dropna(axis=0)

	return data

def run_model(data,mlnmin,mlnmax,target,features,model_type):
	global args
	best_mae = 99999999999
	best_mln_idx = mlnmin

	# Create target object and call it y
	y = data[target]
	# Create features object and call it X
	X = data[features]
	# Split into validation and training data
	train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

	for mln in range(mlnmin,mlnmax,5):
		model = set_model(mln,train_X,train_y,model_type)
		val_predictions = predict_model(model, val_X)
		mae = get_mae(val_predictions,val_y)
		if mae < best_mae:
			best_mae = mae
			best_mln_idx = mln
	print("MLN: "+str(best_mln_idx)+" Mean avg error: "+str(best_mae))


if __name__ == "__main__":
	global args

	parser = argparse.ArgumentParser(description='ML tool')
	add_args(parser)
	args = parser.parse_args()


	data = open_file(args.input)

	if (args.description):
		get_data_description(data)
	print("Model parameters")
	print("----------------------------------------------------------------------------------------------------------------")
	print("Features: " + str(args.features))
	print("target: " + str(args.target))
	print("model type: " + str(args.modeltype))
	print("----------------------------------------------------------------------------------------------------------------")

	run_model(data,int(args.mlnmin),int(args.mlnmax),args.target,args.features,args.modeltype)
	#draw_tree(model,train_X,train_y)
