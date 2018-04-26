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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn import datasets

from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn import tree


from sklearn.model_selection import cross_val_score




#================================================================================
# properties
#================================================================================
dataPath = './data/'


def main():

	fullDf = load_dataframe('DataSet-cleaned-binary.csv')
	trainDf, testDf = splitDataFrame(fullDf, 90)

	# RunDummyClassifier(trainDf, testDf)
	RunDecisionTreeClassifier(trainDf, testDf)
	# RunRandomForestClassifier(trainDf, testDf)
	# RunExtraTreesClassifier(trainDf, testDf)
	# RunAdaBoostClassifier(trainDf, testDf)
	# RunBaggingClassifier(trainDf, testDf)
	# RunGradientBoostingClassifier(trainDf, testDf)
	# RunVotingClassifier(trainDf, testDf)
	# RunKNeighborsClassifier(trainDf, testDf)

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
	train_X = train.drop(['Cover_Type'], axis=1).values		# training set (sample)
	train_y = train.Cover_Type.values						# target feature (to predict)
	test_X = test.drop(['Cover_Type'], axis=1).values

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	dc = DummyClassifier(strategy='stratified', random_state=0)
	dc.fit(X, y)			# Train
	y_dc = dc.predict(X_)	# Predict / y_dc represents the estimated targets as returned by our classifier 

	evaluateModel(y_, y_dc)	# Evaluating model with validation set

	dc.fit(train_X, train_y)		# Retrain with entire training set
	y_test_dc = dc.predict(test_X)	# Predict with test set

	# Write to CSV
	pd.DataFrame({'Cover_Type': y_test_dc}).sort_index(ascending=False, axis=1).to_csv('./data/predictions/dc1.csv', index=False)


def RunDecisionTreeClassifier(train, test):
	# Create numpy arrays for use with scikit-learn
	train_X = train.drop(['Cover_Type'], axis=1).values		# training set (sample)
	train_y = train.Cover_Type.values						# target feature (to predict)
	test_X = test.drop(['Cover_Type'], axis=1).values

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	dtc = tree.DecisionTreeClassifier(criterion='entropy')
	dtc.fit(X, y)				# Train
	y_dtc = dtc.predict(X_)		# Predict / y_dtc represents the estimated targets as returned by our classifier
	
	evaluateModel(y_, y_dtc)	# Evaluating model with validation set

	dtc.fit(train_X, train_y)			# Retrain with entire training set
	y_test_dtc = dtc.predict(test_X)	# Predict with test set

	# Write to CSV
	pd.DataFrame({'Cover_Type': y_test_dtc}).sort_index(ascending=False, axis=1).to_csv('./data/predictions/dtc1.csv', index=False)


def RunRandomForestClassifier(train, test):
	
	# Create numpy arrays for use with scikit-learn
	train_X = train.drop(['Cover_Type'], axis=1).values		# training set (sample)
	train_y = train.Cover_Type.values						# target feature (to predict)
	test_X = test.drop(['Cover_Type'], axis=1).values

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	rfc = RandomForestClassifier(n_estimators=10, criterion='entropy', max_features=None, max_depth=None, verbose=True)
	rfc.fit(X, y)			# Train
	y_rfc = rfc.predict(X_)	# Predict / y_rfc represents the estimated targets as returned by our classifier 

	evaluateModel(y_, y_rfc)	# Evaluating model with validation set

	rfc.fit(train_X, train_y)			# Retrain with entire training set
	y_test_rfc = rfc.predict(test_X)	# Predict with test set

	# Write to CSV
	pd.DataFrame({'Cover_Type': y_test_rfc}).sort_index(ascending=False, axis=1).to_csv('./data/predictions/rf1.csv', index=False)



def RunExtraTreesClassifier(train, test):

	# Create numpy arrays for use with scikit-learn
	train_X = train.drop(['Cover_Type'], axis=1).values		# training set (sample)
	train_y = train.Cover_Type.values						# target feature (to predict)
	test_X = test.drop(['Cover_Type'], axis=1).values

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	etc = ExtraTreesClassifier(n_estimators=10, max_depth=None, random_state=0, verbose=True)
	etc.fit(X, y)			# Train
	y_etc = etc.predict(X_)	# Predict / y_etc represents the estimated targets as returned by our classifier
	
	evaluateModel(y_, y_etc)	# Evaluating model with validation set

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
	
	evaluateModel(y_, y_abc)	# Evaluating model with validation set

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

	bc = BaggingClassifier(KNeighborsClassifier(), max_samples=0.8, max_features=0.8, n_estimators=10)
	bc.fit(X, y)			# Train
	y_bc = bc.predict(X_)	# Predict / y_bc represents the estimated targets as returned by our classifier
	
	evaluateModel(y_, y_bc)	# Evaluating model with validation set

	bc.fit(train_X, train_y)			# Retrain with entire training set
	y_test_bc = bc.predict(test_X)	# Predict with test set

	# Write to CSV
	pd.DataFrame({'Cover_Type': y_test_bc}).sort_index(ascending=False, axis=1).to_csv('./data/predictions/bc1.csv', index=False)


def RunGradientBoostingClassifier(train, test):
	# Create numpy arrays for use with scikit-learn
	train_X = train.drop(['Cover_Type'], axis=1).values		# training set (sample)
	train_y = train.Cover_Type.values						# target feature (to predict)
	test_X = test.drop(['Cover_Type'], axis=1).values

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	gbc = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0)
	gbc.fit(X, y)			# Train
	y_gbc = gbc.predict(X_)	# Predict / y_gbc represents the estimated targets as returned by our classifier
	
	evaluateModel(y_, y_gbc)	# Evaluating model with validation set

	gbc.fit(train_X, train_y)			# Retrain with entire training set
	y_test_bc = gbc.predict(test_X)	# Predict with test set

	# Write to CSV
	pd.DataFrame({'Cover_Type': y_test_bc}).sort_index(ascending=False, axis=1).to_csv('./data/predictions/bc1.csv', index=False)


def RunVotingClassifier(train, test):
	# Create numpy arrays for use with scikit-learn
	train_X = train.drop(['Cover_Type'], axis=1).values		# training set (sample)
	train_y = train.Cover_Type.values						# target feature (to predict)
	test_X = test.drop(['Cover_Type'], axis=1).values

	clf1 = LogisticRegression(random_state=1)
	clf2 = RandomForestClassifier(random_state=1)
	clf3 = GaussianNB()

	eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

	for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):scores = cross_val_score(clf, train_X, train_y, cv=5, scoring='accuracy')
	print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


def RunKNeighborsClassifier(train, test):
	# Create numpy arrays for use with scikit-learn
	train_X = train.drop(['Cover_Type'], axis=1).values		# training set (sample)
	train_y = train.Cover_Type.values						# target feature (to predict)
	test_X = test.drop(['Cover_Type'], axis=1).values

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	knc = KNeighborsClassifier(n_neighbors=1, weights='distance')
	knc.fit(X, y)			# Train
	y_knc = knc.predict(X_)	# Predict / y_knc represents the estimated targets as returned by our classifier
	
	evaluateModel(y_, y_knc)	# Evaluating model with validation set

	knc.fit(train_X, train_y)			# Retrain with entire training set
	y_test_knc = knc.predict(test_X)	# Predict with test set

	# Write to CSV
	pd.DataFrame({'Cover_Type': y_test_knc}).sort_index(ascending=False, axis=1).to_csv('./data/predictions/knc1.csv', index=False)
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


def evaluateModel(y_true, y_pred):
	print('')
	print('Classification report:')
	print(metrics.classification_report(y_true, y_pred))
	print('')
	print('Confusion matrix:')
	print(metrics.confusion_matrix(y_true, y_pred))
	print('')
	print('Accuracy score: ' + str(metrics.accuracy_score(y_true, y_pred)))
	print('')
	return None


def dataframeToNumpy(df):
	return df[df.columns.values].values


def splitDataFrame(fullDf, trainPercentage):
	test_size = (100 - trainPercentage) / 100
	trainDf, testDf = train_test_split(fullDf, test_size=test_size)

	# print('Original: ' + str(fullDf.size) + ' / ' + str(fullDf.shape))
	# print('Train: ' + str(trainDf.size) + ' / ' + str(trainDf.shape))
	# print('Test: ' + str(testDf.size) + ' / ' + str(testDf.shape))
	return trainDf, testDf

# program launch
if __name__ == '__main__':
	main()