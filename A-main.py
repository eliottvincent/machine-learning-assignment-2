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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn import datasets

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
	# RunDecisionTreeClassifier(trainDf, testDf)
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


def trainAndRetrainClassifier(classifier, X, X_, y, y_, train_X, train_y, test_X):
	"""trainAndRetrainClassifier
	Trains a classifier a first time, predicts a target feature and train the classifier again thereafter.

    Input:
    classifier -- the classifier to use
    X -- X split of training set
    X_ -- X split of testing set
    y -- y split of training set
    y_ -- y splif of testing set
    train_X -- training set (sample)
    train_y -- target feature
    test_X -- testing set

	Output:
	y_test_c -- predicted values
    """

	classifier.fit(X, y)			# Training a first time
	y_c = classifier.predict(X_)	# Predicting (y_c represents the estimated targets as returned by the classifier)

	evaluateModel(y_, y_c)			# Evaluating model with validation set

	classifier.fit(train_X, train_y)		# Training again (with entire training set)
	y_test_c = classifier.predict(test_X)	# Predicting with test set

	return y_test_c


def RunDummyClassifier(trainDf, testDf):
	"""RunDummyClassifier
	Runs a Dummy Classifier on training and testing dataframes.

    Input:
    trainDf -- the training DataFrame (pandas)
    testDf -- the testing DataFrame (pandas)
    """

	train_X, train_y, test_X = createArrays(trainDf, testDf)
	
	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	dc = DummyClassifier(strategy='stratified', random_state=0)
	
	y_test_dc = trainAndRetrainClassifier(dc, X, X_, y, y_, train_X, train_y, test_X)

	savePredictions(y_test_dc, 'dc.csv')


def RunDecisionTreeClassifier(trainDf, testDf):
	"""RunDecisionTreeClassifier
	Runs a Decision Tree Classifier on training and testing dataframes.

    Input:
    trainDf -- the training DataFrame (pandas)
    testDf -- the testing DataFrame (pandas)
    """
	train_X, train_y, test_X = createArrays(trainDf, testDf)

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	dtc = tree.DecisionTreeClassifier(criterion='entropy')

	y_test_dtc = trainAndRetrainClassifier(dtc, X, X_, y, y_, train_X, train_y, test_X)
	
	savePredictions(y_test_dtc, 'dtc.csv')
	

def RunRandomForestClassifier(trainDf, testDf):
	"""RunRandomForestClassifier
	Runs a Random Forest Classifier on training and testing dataframes.

    Input:
    trainDf -- the training DataFrame (pandas)
    testDf -- the testing DataFrame (pandas)
    """
	train_X, train_y, test_X = createArrays(trainDf, testDf)

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	rfc = RandomForestClassifier(n_estimators=10, criterion='entropy', max_features=None, max_depth=None, verbose=True)
	
	y_test_rfc = trainAndRetrainClassifier(rfc, X, X_, y, y_, train_X, train_y, test_X)
	
	savePredictions(y_test_rfc, 'rfc.csv')
	

def RunExtraTreesClassifier(trainDf, testDf):
	"""RunExtraTreesClassifier
	Runs a Extra Trees Classifier on training and testing dataframes.

    Input:
    trainDf -- the training DataFrame (pandas)
    testDf -- the testing DataFrame (pandas)
    """
	train_X, train_y, test_X = createArrays(trainDf, testDf)

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	etc = ExtraTreesClassifier(n_estimators=10, max_depth=None, random_state=0, verbose=True)

	y_test_etc = trainAndRetrainClassifier(etc, X, X_, y, y_, train_X, train_y, test_X)
	
	savePredictions(y_test_etc, 'etc.csv')
	

def RunAdaBoostClassifier(trainDf, testDf):
	"""RunAdaBoostClassifier
	Runs an Ada Boost Classifier on training and testing dataframes.

    Input:
    trainDf -- the training DataFrame (pandas)
    testDf -- the testing DataFrame (pandas)
    """
	train_X, train_y, test_X = createArrays(trainDf, testDf)

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	abc = AdaBoostClassifier(n_estimators=10)

	y_test_abc = trainAndRetrainClassifier(abc, X, X_, y, y_, train_X, train_y, test_X)
	
	savePredictions(y_test_abc, 'abc.csv')


def RunBaggingClassifier(trainDf, testDf):
	"""RunBaggingClassifier
	Runs a Bagging Classifier on training and testing dataframes.

    Input:
    trainDf -- the training DataFrame (pandas)
    testDf -- the testing DataFrame (pandas)
    """
	train_X, train_y, test_X = createArrays(trainDf, testDf)

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	bc = BaggingClassifier(KNeighborsClassifier(), max_samples=0.8, max_features=0.8, n_estimators=10)
	
	y_test_bc = trainAndRetrainClassifier(bc, X, X_, y, y_, train_X, train_y, test_X)
	
	savePredictions(y_test_bc, 'bc.csv')


def RunGradientBoostingClassifier(trainDf, testDf):
	"""RunGradientBoostingClassifier
	Runs a Gradient Boosting Classifier on training and testing dataframes.

    Input:
    trainDf -- the training DataFrame (pandas)
    testDf -- the testing DataFrame (pandas)
    """
	train_X, train_y, test_X = createArrays(trainDf, testDf)

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	gbc = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0)

	y_test_gbc = trainAndRetrainClassifier(gbc, X, X_, y, y_, train_X, train_y, test_X)
	
	savePredictions(y_test_gbc, 'bc.csv')


def RunVotingClassifier(trainDf, testDf):
	"""RunVotingClassifier
	Runs a Voting Classifier on training and testing dataframes.

    Input:
    trainDf -- the training DataFrame (pandas)
    testDf -- the testing DataFrame (pandas)
    """
	train_X, train_y, test_X = createArrays(trainDf, testDf)

	clf1 = LogisticRegression(random_state=1)
	clf2 = RandomForestClassifier(random_state=1)
	clf3 = GaussianNB()

	eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

	for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):scores = cross_val_score(clf, train_X, train_y, cv=5, scoring='accuracy')
	print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


def RunKNeighborsClassifier(trainDf, testDf):
	"""RunKNeighborsClassifier
	Runs a K-Neighbors Classifier on training and testing dataframes.

    Input:
    trainDf -- the training DataFrame (pandas)
    testDf -- the testing DataFrame (pandas)
    """
	train_X, train_y, test_X = createArrays(trainDf, testDf)

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	knc = KNeighborsClassifier(n_neighbors=1, weights='distance')
	
	y_test_knc = trainAndRetrainClassifier(knc, X, X_, y, y_, train_X, train_y, test_X)
	
	savePredictions(y_test_knc, 'knc.csv')




#  ███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗     
#  ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║     
#  ██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║     
#  ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║     
#  ██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗
#  ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝


def RunMLPClassifier(trainDf, testDf):
	"""RunMLPClassifier
	Runs a Neural Network based on a MLP Classifier on training and testing dataframes.

    Input:
    trainDf -- the training DataFrame (pandas)
    testDf -- the testing DataFrame (pandas)
    """
	train_X, train_y, test_X = createArrays(trainDf, testDf)

	# Split the training set into training and validation sets
	X, X_, y, y_ = train_test_split(train_X, train_y, test_size=0.2)

	mlpc = MLPClassifier(verbose=True)
	mlpc.fit(X, y)			# Train
	MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=True,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='adaptive',
       learning_rate_init=0.001, max_iter=400, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=1e-7, validation_fraction=0.1, verbose=True,
       warm_start=False)
	y_mlpc = mlpc.predict(X_)	# Predict / y_rf represents the estimated targets as returned by our classifier 

	evaluateModel(y_, y_mlpc)	# Evaluating model with validation set

	mlpc.fit(train_X, train_y)		# Retrain with entire training set
	y_test_mlpc = mlpc.predict(test_X)	# Predict with test set




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


def savePredictions(predictions, fileName):
	"""savePredictions
	Saves predictions into a .csv file.

    Input:
    predictions -- predictions to save
    fileName -- the name of the file (will be saved in ./data/predictions/)
    """
	pd.DataFrame({'Cover_Type': predictions}).sort_index(ascending=False, axis=1).to_csv('./data/predictions/' + fileName, index=False)




#  ██╗   ██╗████████╗██╗██╗     ███████╗
#  ██║   ██║╚══██╔══╝██║██║     ██╔════╝
#  ██║   ██║   ██║   ██║██║     ███████╗
#  ██║   ██║   ██║   ██║██║     ╚════██║
#  ╚██████╔╝   ██║   ██║███████╗███████║
#   ╚═════╝    ╚═╝   ╚═╝╚══════╝╚══════╝


def createArrays(trainDf, testDf):
	"""createArrays
	Saves predictions into a .csv file.

    Input:
    trainDf -- the training DataFrame (pandas)
    testDf -- the testing DataFrame (pandas)

    Output:
    train_X -- training set (sample)
    train_y -- target feature
    test_X -- testing set
    """
	train_X = trainDf.drop(['Cover_Type'], axis=1).values	# training set (sample)
	train_y = trainDf.Cover_Type.values						# target feature (to predict)
	test_X = testDf.drop(['Cover_Type'], axis=1).values

	return train_X, train_y, test_X


def getFolds(n_splits, trainDf):
	"""getFolds
	Saves predictions into a .csv file.

    Input:
    n_splits -- the number of splits to perform
    trainDf -- the dataframe to split
    """

	kf = KFold(n_splits=n_splits, random_state=None)

	for train_index, test_index in kf.split(X=trainDf):

		print("TRAIN:", train_index, "TEST:", test_index)
		
		X = trainDf.values
		y = trainDf.Cover_Type.values
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]


def evaluateModel(y_true, y_pred):
	"""evaluateModel
	Evaluates a model based on several measures.

    Input:
    y_true -- the valid target values
    y_pred -- the predicted values
    """
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


def splitDataFrame(fullDf, trainPercentage):
	"""splitDataFrame
	Splits a dataframe into a training dataframe and a testing dataframe.

    Input:
    fullDf -- the dataframe to split
    trainPercentage -- the percentage of space to accord to the test set

    Output:
	trainDf -- the training dataframe
	testDf -- the testing dataframe
    """
	test_size = (100 - trainPercentage) / 100
	trainDf, testDf = train_test_split(fullDf, test_size=test_size)

	# print('Original: ' + str(fullDf.size) + ' / ' + str(fullDf.shape))
	# print('Train: ' + str(trainDf.size) + ' / ' + str(trainDf.shape))
	# print('Test: ' + str(testDf.size) + ' / ' + str(testDf.shape))
	return trainDf, testDf

# program launch
if __name__ == '__main__':
	main()