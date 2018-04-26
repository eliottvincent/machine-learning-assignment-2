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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


#================================================================================
# properties
#================================================================================
dataPath = './data/'
featuresNames = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area', 'Soil_Type', 'Cover_Type']

def main():

	fullDf = load_dataframe('DataSet-cleaned-integer.csv')
	trainDf, testDf = splitDataFrame(fullDf, 90)

	# convertedDf = dataframeToNumpy(df)

	RunRandomForestClassifier(fullDf, trainDf, testDf)
	# RunExtraTreesClassifier(trainDf, testDf)


	return None




#   ██████╗██╗      █████╗ ███████╗███████╗██╗███████╗██╗███████╗██████╗ ███████╗
#  ██╔════╝██║     ██╔══██╗██╔════╝██╔════╝██║██╔════╝██║██╔════╝██╔══██╗██╔════╝
#  ██║     ██║     ███████║███████╗███████╗██║█████╗  ██║█████╗  ██████╔╝███████╗
#  ██║     ██║     ██╔══██║╚════██║╚════██║██║██╔══╝  ██║██╔══╝  ██╔══██╗╚════██║
#  ╚██████╗███████╗██║  ██║███████║███████║██║██║     ██║███████╗██║  ██║███████║
#   ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝


def RunRandomForestClassifier(fullDf, trainDf, testDf):
	
	# Load the training and test data sets
	train = load_dataframe('DataSet-cleaned-integer.csv')
	test = load_dataframe('DataSet-1000-cleaned.csv')

	# Create numpy arrays for use with scikit-learn
	# train_X = train.drop(['Id','Cover_Type'],axis=1).values
	train_X = train.drop(['Cover_Type'], axis=1).values	# sample
	train_y = train.Cover_Type.values						# target feature (to predict)
	# test_X = test.drop('Id',axis=1).values
	test_X = test.values

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	# Train and predict with the random forest classifier
	rf = RandomForestClassifier()
	rf.fit(X, y)
	y_rf = rf.predict(X_)
	print(metrics.classification_report(y_, y_rf))
	print(metrics.accuracy_score(y_, y_rf))

	# Retrain with entire training set and predict test set.
	rf.fit(train_X, train_y)
	y_test_rf = rf.predict(test_X)

	# Write to CSV
	fullDf.DataFrame({'Cover_Type': y_test_rf}).sort_index(ascending=False, axis=1).to_csv('rf1.csv', index=False)
	return None


def RunExtraTreesClassifier():
	et = ExtraTreesClassifier(n_estimators=100, max_depth=None, random_state=0)

	columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
	columnsBis = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology']

	labels = df['Cover_Type'].values
	features = df[list(columnsBis)].values	# getting all features
	
	et_score = cross_val_score(et, features, labels, n_jobs=-1).mean()

	print("{0} -> ET: {1})".format(columns, et_score))




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
	return pd.read_csv(path, header=None, names=featuresNames)


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