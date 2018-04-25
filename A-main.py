__author__ = "Simon Bonnaud, Arthur Chevallier, Anaïs Pignet, Eliott Vincent"
__license__ = "MIT"
__version__ = "0.1"

#================================================================================
# modules
#================================================================================
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score


#================================================================================
# properties
#================================================================================
dataPath = './data/'
featuresNames = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area', 'Soil_Type', 'Cover_Type']

def main():
	df = load_dataframe('DataSet-cleaned-integer.csv')
	print(df)
	return None




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


def dataframeToNumpy(df):
	return df[df.columns.values].values


# program launch
if __name__ == '__main__':
	main()