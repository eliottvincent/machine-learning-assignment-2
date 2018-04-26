__author__ = "Simon Bonnaud, Arthur Chevallier, Anaïs Pignet, Eliott Vincent"
__license__ = "MIT"
__version__ = "0.1"

#================================================================================
# modules
#================================================================================
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn import tree


#================================================================================
# properties
#================================================================================
dataPath = './data/'


def main():

	fullDf = load_dataframe('DataSet-cleaned-binary.csv')
	trainDf, testDf = splitDataFrame(fullDf, 90)

	# RunDummyClassifier(trainDf, testDf)
	# RunRandomForestClassifier(trainDf, testDf)
	# RunDecisionTreeClassifier(trainDf, testDf)
	# RunExtraTreesClassifier(trainDf, testDf)
	# RunAdaBoostClassifier(trainDf, testDf)
	# RunBaggingClassifier(trainDf, testDf)
	RunGradientBoostingClassifier(trainDf, testDf)



	# RunMLPClassifier(trainDf, testDf)
	
	return None




#   ██████╗██╗      █████╗ ███████╗███████╗██╗███████╗██╗███████╗██████╗ ███████╗
#  ██╔════╝██║     ██╔══██╗██╔════╝██╔════╝██║██╔════╝██║██╔════╝██╔══██╗██╔════╝
#  ██║     ██║     ███████║███████╗███████╗██║█████╗  ██║█████╗  ██████╔╝███████╗
#  ██║     ██║     ██╔══██║╚════██║╚════██║██║██╔══╝  ██║██╔══╝  ██╔══██╗╚════██║
#  ╚██████╗███████╗██║  ██║███████║███████║██║██║     ██║███████╗██║  ██║███████║
#   ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝



def RunDummyClassifier(train, test):
	
	# Create numpy arrays for use with scikit-learn
	# train_X = train.drop(['Id','Cover_Type'],axis=1).values
	train_X = train.drop(['Cover_Type'], axis=1).values		# training set (sample)
	train_y = train.Cover_Type.values						# target feature (to predict)
	# test_X = test.drop('Id',axis=1).values
	test_X = test.drop(['Cover_Type'], axis=1).values

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	dc = DummyClassifier(strategy='most_frequent', random_state=0)
	dc.fit(X, y)			# Train
	y_dc = dc.predict(X_)	# Predict / y_dc represents the estimated targets as returned by our classifier 

	# Evaluating model with validation set
	print(metrics.classification_report(y_, y_dc))
	print(metrics.confusion_matrix(y_, y_dc))
	print(metrics.accuracy_score(y_, y_dc))
	print(metrics.r2_score(y_, y_dc))

	dc.fit(train_X, train_y)		# Retrain with entire training set
	y_test_dc = dc.predict(test_X)	# Predict with test set

	# Write to CSV
	pd.DataFrame({'Cover_Type': y_test_dc}).sort_index(ascending=False, axis=1).to_csv('./data/predictions/dc1.csv', index=False)


def RunRandomForestClassifier(train, test):
	
	# Create numpy arrays for use with scikit-learn
	# train_X = train.drop(['Id','Cover_Type'],axis=1).values
	train_X = train.drop(['Cover_Type'], axis=1).values		# training set (sample)
	train_y = train.Cover_Type.values						# target feature (to predict)
	# test_X = test.drop('Id',axis=1).values
	test_X = test.drop(['Cover_Type'], axis=1).values

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	rfc = RandomForestClassifier(n_estimators=10, max_depth=None, verbose=True)
	rfc.fit(X, y)			# Train
	y_rfc = rfc.predict(X_)	# Predict / y_rfc represents the estimated targets as returned by our classifier 

	# Evaluating model with validation set
	print(metrics.classification_report(y_, y_rfc))
	print(metrics.confusion_matrix(y_, y_rfc))
	print(metrics.accuracy_score(y_, y_rfc))
	print(metrics.r2_score(y_, y_rfc))

	rfc.fit(train_X, train_y)			# Retrain with entire training set
	y_test_rfc = rfc.predict(test_X)	# Predict with test set

	# Write to CSV
	pd.DataFrame({'Cover_Type': y_test_rfc}).sort_index(ascending=False, axis=1).to_csv('./data/predictions/rf1.csv', index=False)


def RunDecisionTreeClassifier(train, test):
	# Create numpy arrays for use with scikit-learn
	train_X = train.drop(['Cover_Type'], axis=1).values		# training set (sample)
	train_y = train.Cover_Type.values						# target feature (to predict)
	test_X = test.drop(['Cover_Type'], axis=1).values

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	dtc = tree.DecisionTreeClassifier()
	dtc.fit(X, y)				# Train
	y_dtc = dtc.predict(X_)		# Predict / y_dtc represents the estimated targets as returned by our classifier
	
	# Evaluating model with validation set
	print(metrics.classification_report(y_, y_dtc))
	print(metrics.confusion_matrix(y_, y_dtc))
	print(metrics.accuracy_score(y_, y_dtc))
	print(metrics.r2_score(y_, y_dtc))

	dtc.fit(train_X, train_y)			# Retrain with entire training set
	y_test_dtc = dtc.predict(test_X)	# Predict with test set

	# Write to CSV
	pd.DataFrame({'Cover_Type': y_test_dtc}).sort_index(ascending=False, axis=1).to_csv('./data/predictions/dtc1.csv', index=False)


def RunExtraTreesClassifier(train, test):

	# Create numpy arrays for use with scikit-learn
	train_X = train.drop(['Cover_Type'], axis=1).values		# training set (sample)
	train_y = train.Cover_Type.values						# target feature (to predict)
	test_X = test.drop(['Cover_Type'], axis=1).values

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	etc = ExtraTreesClassifier(n_estimators=100, max_depth=None, random_state=0, verbose=True)
	etc.fit(X, y)			# Train
	y_etc = etc.predict(X_)	# Predict / y_etc represents the estimated targets as returned by our classifier
	
	# Evaluating model with validation set
	print(metrics.classification_report(y_, y_etc))
	print(metrics.confusion_matrix(y_, y_etc))
	print(metrics.accuracy_score(y_, y_etc))
	print(metrics.r2_score(y_, y_etc))

	etc.fit(train_X, train_y)		# Retrain with entire training set
	y_test_etc = etc.predict(test_X)	# Predict with test set

	# Write to CSV
	pd.DataFrame({'Cover_Type': y_test_etc}).sort_index(ascending=False, axis=1).to_csv('./data/predictions/etc1.csv', index=False)


def RunAdaBoostClassifier(train, test):

	# Create numpy arrays for use with scikit-learn
	train_X = train.drop(['Cover_Type'], axis=1).values		# training set (sample)
	train_y = train.Cover_Type.values						# target feature (to predict)
	test_X = test.drop(['Cover_Type'], axis=1).values

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	abc = AdaBoostClassifier(n_estimators=10)
	abc.fit(X, y)			# Train
	y_abc = abc.predict(X_)	# Predict / y_abc represents the estimated targets as returned by our classifier
	
	# Evaluating model with validation set
	print(metrics.classification_report(y_, y_abc))
	print(metrics.confusion_matrix(y_, y_abc))
	print(metrics.accuracy_score(y_, y_abc))
	print(metrics.r2_score(y_, y_abc))

	abc.fit(train_X, train_y)			# Retrain with entire training set
	y_test_abc = abc.predict(test_X)	# Predict with test set

	# Write to CSV
	pd.DataFrame({'Cover_Type': y_test_abc}).sort_index(ascending=False, axis=1).to_csv('./data/predictions/abc1.csv', index=False)


def RunBaggingClassifier(train, test):

	# Create numpy arrays for use with scikit-learn
	train_X = train.drop(['Cover_Type'], axis=1).values		# training set (sample)
	train_y = train.Cover_Type.values						# target feature (to predict)
	test_X = test.drop(['Cover_Type'], axis=1).values

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	bc = BaggingClassifier(KNeighborsClassifier(), n_estimators=10)
	bc.fit(X, y)			# Train
	y_bc = bc.predict(X_)	# Predict / y_bc represents the estimated targets as returned by our classifier
	
	# Evaluating model with validation set
	print(metrics.classification_report(y_, y_bc))
	print(metrics.confusion_matrix(y_, y_bc))
	print(metrics.accuracy_score(y_, y_bc))
	print(metrics.r2_score(y_, y_bc))

	bc.fit(train_X, train_y)			# Retrain with entire training set
	y_test_bc = bc.predict(test_X)	# Predict with test set

	# Write to CSV
	pd.DataFrame({'Cover_Type': y_test_bc}).sort_index(ascending=False, axis=1).to_csv('./data/predictions/bc1.csv', index=False)

#  ██╗███╗   ██╗██████╗ ██╗   ██╗████████╗    ██████╗ ██╗   ██╗████████╗██████╗ ██╗   ██╗████████╗
#  ██║████╗  ██║██╔══██╗██║   ██║╚══██╔══╝   ██╔═══██╗██║   ██║╚══██╔══╝██╔══██╗██║   ██║╚══██╔══╝
#  ██║██╔██╗ ██║██████╔╝██║   ██║   ██║█████╗██║   ██║██║   ██║   ██║   ██████╔╝██║   ██║   ██║   
#  ██║██║╚██╗██║██╔═══╝ ██║   ██║   ██║╚════╝██║   ██║██║   ██║   ██║   ██╔═══╝ ██║   ██║   ██║   
#  ██║██║ ╚████║██║     ╚██████╔╝   ██║      ╚██████╔╝╚██████╔╝   ██║   ██║     ╚██████╔╝   ██║   
#  ╚═╝╚═╝  ╚═══╝╚═╝      ╚═════╝    ╚═╝       ╚═════╝  ╚═════╝    ╚═╝   ╚═╝      ╚═════╝    ╚═╝   


def load_dataframe(fileName):
	"""load_dataframe
	Loads a DataFrame from a .csv file.

    Input:
    fileName -- the name of the file (should be located in ./data/)
    
	Output:
	pd.read_csv() -- the DataFrame loaded by Pandas
    """
	path = dataPath + fileName
	return pd.read_csv(path, header=0)


def write_dataframe(df, fileName):
	"""write_dataframe
	Writes a DataFrame into a .csv file.

    Input:
    df -- the DataFrame to write
    fileName -- the name of the file (will be saved in ./data/)
    """
	path = dataPath + fileName
	df.to_csv(path)




#  ██╗   ██╗████████╗██╗██╗     ███████╗
#  ██║   ██║╚══██╔══╝██║██║     ██╔════╝
#  ██║   ██║   ██║   ██║██║     ███████╗
#  ██║   ██║   ██║   ██║██║     ╚════██║
#  ╚██████╔╝   ██║   ██║███████╗███████║
#   ╚═════╝    ╚═╝   ╚═╝╚══════╝╚══════╝


def dataframeToNumpy(df):
	return df[df.columns.values].values


def splitDataFrame(fullDf, trainPercentage):
	test_size = (100 - trainPercentage) / 100
	trainDf, testDf = train_test_split(fullDf, test_size=test_size)

	print('Original: ' + str(fullDf.size) + ' / ' + str(fullDf.shape))
	print('Train: ' + str(trainDf.size) + ' / ' + str(trainDf.shape))
	print('Test: ' + str(testDf.size) + ' / ' + str(testDf.shape))
	return trainDf, testDf

# program launch
if __name__ == '__main__':
	main()