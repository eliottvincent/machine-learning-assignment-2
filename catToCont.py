import pandas as pd


#================================================================================
# properties
#================================================================================
featuresNames = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area', 'Soil_Type', 'Cover_Type']
cols_to_transform = ['Wilderness_Area', 'Soil_Type']
dataPath = './data/'

def catToCont():
	
	fullDf = load_dataframe('DataSet-cleaned-integer.csv')
	print(fullDf)
	df_with_dummies = pd.get_dummies(data = fullDf, columns = cols_to_transform)
	print(df_with_dummies)
	write_dataframe(df_with_dummies, 'DataSet-cleaned-continuous.csv')
	



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

catToCont()