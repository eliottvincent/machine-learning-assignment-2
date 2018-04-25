__author__ = "Simon Bonnaud, Arthur Chevallier, Anaïs Pignet, Eliott Vincent"
__license__ = "MIT"
__version__ = "0.1"

#================================================================================
# modules
#================================================================================
import pandas as pd
import numpy as np
import plotly
from plotly.graph_objs import Scatter, Layout
import plotly.graph_objs as go
from collections import Counter


#================================================================================
# properties
#================================================================================
featuresNames = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area', 'Soil_Type', 'Cover_Type']
continuousFeatures = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
categoricalFeatures = ['Wilderness_Area', 'Soil_Type', 'Cover_Type']
continuousStatistics = ['FEATURENAME', 'count', 'miss_percentage', 'card', 'minimum', 'first_quartile', 'mean', 'median', 'third_quartile', 'maximum', 'std_dev']
categoricalStatistics = ['FEATURENAME', 'count', 'miss_percentage', 'card', 'mode', 'mode_frequency', 'mode_percentage', 'second_mode', 'second_mode_frequency', 'second_mode_percentage']
dataPath = './data/'
teamName = 'A'


def main():
	# loading the dataset as a Pandas DataFrame
	df = load_dataframe('DataSet-cleaned.csv')

	# splitting the main DataFrame in two sub-dataframes (continuous and categorical features)
	continuousDf = getContinuousDf(df)
	categoricalDf = getCategoricalDf(df)

	# generating the reports for bot continuous and categorical features
	continuousReport = generateContinuousReport(continuousDf)
	categoricalReport = generateCategoricalReport(categoricalDf)
	
	# saving the reports as .csv files
	write_dataframe(continuousReport, teamName + '-DQR-ContinuousFeatures.csv')
	write_dataframe(categoricalReport, teamName + '-DQR-CategoricalFeatures.csv')

	# generating the graphs for continuous and categorical features
	generateContinuousGraphs(continuousDf, continuousReport)
	generateCategoricalGraphs(categoricalDf)




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


def getContinuousDf(df):
	"""getContinuousDf
	Extract continuous features from the main DataFrame.

    Input:
    df -- the main DataFrame
    
	Output:
	df.drop() -- the main DataFrame with categoricalFeatures omitted
    """
	return df.drop(categoricalFeatures, axis=1)


def getCategoricalDf(df):
	"""getCategoricalDf
	Extract categorical features from the main DataFrame.

    Input:
    df -- the main DataFrame
    
	Output:
	df.drop() -- the main DataFrame with continuousFeatures omitted
    """
	return df.drop(continuousFeatures, axis=1)




#  ███████╗████████╗ █████╗ ████████╗██╗███████╗████████╗██╗ ██████╗███████╗
#  ██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝██║██╔════╝╚══██╔══╝██║██╔════╝██╔════╝
#  ███████╗   ██║   ███████║   ██║   ██║███████╗   ██║   ██║██║     ███████╗
#  ╚════██║   ██║   ██╔══██║   ██║   ██║╚════██║   ██║   ██║██║     ╚════██║
#  ███████║   ██║   ██║  ██║   ██║   ██║███████║   ██║   ██║╚██████╗███████║
#  ╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚═╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝╚══════╝


def computeContinuousStatistics(df):
	"""computeContinuousStatistics
	Computes statistics for continuous features.

    Input:
    df -- DataFrame to use
    
	Output:
	{} -- a dictionary containing the statistics
    """

	# returning the final statistics
	return {
		'count': len(df),
		'miss_percentage': ((len(df)-df.count())*100)/len(df),
		'card': df.nunique(),
		'minimum': df.min(),
		'first_quartile': df.quantile([0.25][0]),
		'mean': df.mean(),
		'median': df.median(),
		'third_quartile': df.quantile([0.75][0]),
		'maximum': df.max(),
		'std_dev': df.std()
	}


def computeCategoricalStatistics(df):
	"""computeCategoricalStatistics
	Computes statistics for categorical features.

    Input:
    df -- DataFrame containing to use
    
	Output:
	{} -- a dictionary containing the statistics
    """
	
	# computing first mode
	tmpDf = df.copy()
	firstMode = tmpDf.mode()[0]
	firstModeFrequency = tmpDf.describe(include='all')['freq']
	firstModePercentage = (firstModeFrequency/len(df))*100
	
	# computing second mode
	tmpDf = tmpDf[tmpDf != firstMode]	# removing rows with first mode values
	secondMode = tmpDf.mode()[0]
	secondModeFrequency = tmpDf.describe(include='all')['freq']
	secondModePercentage = (secondModeFrequency/len(df))*100

	# returning the final statistics
	return {
		'count': len(df),
		'miss_percentage': ((len(df)-df.count())*100)/len(df),
		'card': df.nunique(),
		'mode': firstMode,
		'mode_frequency': firstModeFrequency,
		'mode_percentage': firstModePercentage,
		'second_mode': secondMode,
		'second_mode_frequency': secondModeFrequency,
		'second_mode_percentage': secondModePercentage
	}




#  ██████╗ ███████╗██████╗  ██████╗ ██████╗ ████████╗███████╗
#  ██╔══██╗██╔════╝██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔════╝
#  ██████╔╝█████╗  ██████╔╝██║   ██║██████╔╝   ██║   ███████╗
#  ██╔══██╗██╔══╝  ██╔═══╝ ██║   ██║██╔══██╗   ██║   ╚════██║
#  ██║  ██║███████╗██║     ╚██████╔╝██║  ██║   ██║   ███████║
#  ╚═╝  ╚═╝╚══════╝╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝


def generateContinuousReport(continuousDf):
	"""generateContinuousReport
	Generates a report of continuous features statistics from a dataframe.

    Input:
    continuousDf -- DataFrame to use
    
	Output:
	generateReport() -- output of the generateReport() method
    """
	report = generateReport(continuousDf, continuousStatistics, computeContinuousStatistics)
	print('Report for continuous features: OK')
	return report


def generateCategoricalReport(categoricalDf):
	"""generateCategoricalReport
	Generates a report of categorical features statistics from a dataframe.

    Input:
    categoricalDf -- DataFrame to use
    
	Output:
	generateReport() -- output of the generateReport() method
    """
	report = generateReport(categoricalDf, categoricalStatistics, computeCategoricalStatistics)
	print('Report for categorical features: OK')
	return report


def generateReport(dataFrame, statisticsNames, computeFunction):
	"""generateReport
	Generates a report from a dataframe.

    Input:
    dataFrame -- DataFrame to use
    statisticsNames -- statistics to compute
    computeFunction -- method to use to compute the mesures

	Output:
	statisticsDf -- DataFrame containing the statistics
    """

	# 1st step: create an empty dataframe with the statistics names as columns
	statisticsDf = pd.DataFrame(columns=statisticsNames)
	statisticsDf.set_index('FEATURENAME', inplace=True)

	# 2nd step: loop over each feature
	for featureName in dataFrame:
		
		# gathering all values for the current feature
		featureValues = dataFrame[featureName]

		# computing statistics
		statisticsDict = computeFunction(featureValues)

		# creating a serie with those statistics
		statisticsSeries = pd.Series(statisticsDict, name=featureName)

		# adding the serie to the final dataframe
		statisticsDf = statisticsDf.append(statisticsSeries)

	# 3rd step: return the completed dataframe
	return statisticsDf




#   ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗███████╗
#  ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║██╔════╝
#  ██║  ███╗██████╔╝███████║██████╔╝███████║███████╗
#  ██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║╚════██║
#  ╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║███████║
#   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝


def drawHistogramFromFeature(featureName, featureValues):
	"""drawHistogramFromFeature
	Draws an histogram graph from a feature.
	The graph is then saved in ./data/

    Input:
    featureName -- the name of the feature
    featureValues -- the values of the feature
    """

	data = [go.Histogram(x=featureValues)]
	path = dataPath + featureName + '-histogram.html'
	plotly.offline.plot(data, filename=path, auto_open=False)


def drawBarPlotFromFeature(featureName, featureValues):
	"""drawBarPlotFromFeature
	Draws a bar plot graph from a feature.
	The graph is then saved in ./data/

    Input:
    featureName -- the name of the feature
    featureValues -- the values of the feature
    """

    # we need to count the number of each occurrence
	occurrences = Counter(featureValues)
	x_axis = []
	y_axis = []
	for occurrence in occurrences:
		x_axis.append(occurrence)
		y_axis.append(occurrences[occurrence])

	data = [go.Bar(x=x_axis, y=y_axis)]
	path = dataPath + featureName + '-bar-plot.html'
	plotly.offline.plot(data, filename=path, auto_open=False)


def generateContinuousGraphs(continuousDf, continuousReport):
	"""generateContinuousGraphs
	Draws graphs for each continuous feature of a DataFrame.
	According to the cardinality of the feature, the generated graph will either be an histogram or a bar plot.

    Input:
    continuousDf -- the DataFrame containing all the continuous features
    continuousReport -- the report associated with the continuous features
    """

	for featureName in continuousDf:

		cardinality = continuousReport.loc[featureName]['card']	# getting the cardinality from the report

		# if the continuous feature has a low cardinality (< 10), we draw a bar plot
		if cardinality < 10:
			drawBarPlotFromFeature(featureName, continuousDf[featureName])
		# else we draw an histogram
		else:
			drawHistogramFromFeature(featureName, continuousDf[featureName])

	print('Graphs for continuous features: OK')


def generateCategoricalGraphs(categoricalDf):
	"""generateCategoricalGraphs
	Draws graphs for each categorical feature of a DataFrame.

    Input:
    categoricalDf -- the DataFrame containing all the categorical features
    """

	for featureName in categoricalDf:
		drawBarPlotFromFeature(featureName, categoricalDf[featureName])

	print('Graphs for categorical features: OK')


# program launch
if __name__ == '__main__':
	main()