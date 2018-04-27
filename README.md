#############################################################################
# Project : machine-learning-assignment-2                                   #
# Autor : Simon Bonnaud - Arthur Chevallier - Anaïs Pignet - Eliott Vincent #
# Subject : machine-learning												#
#############################################################################

#################### DESCRIPTION ####################
This program is designed to load a set of data from a 
CSV file. The code then generates:
* Implements classifiers models
* Use the data set to train and test on the models
# Generate evaluation measurements

#################### COMPOSITION ####################
Files ->
* .gitignore: a git ignore file for versioning purpose
* A-main.py: the main file
* catToBin.py: convert binary features into sub-features
* correct.py: convert data into useful data
* dqr.py: generate Data Quality Reports from the dataset
* LICENSE: the software license to tell others what they can and can't do with our source code
* README.md: the present file
Folders ->
* ./data/: the folder containing the data set (DataSet.csv) and the generated files

################### ENVIRONNEMENT ###################
Python: 3.6.4
Modules: pandas, numpy and sklearn

##################### EXECUTION #####################
In the main file (A-main.py), you need to uncomment the
fonction corresponding to the model you want to try
command: python A-main.py

################# OUTPUTS FORMATS ###################
Data Quality Reports: CSV format
Histograms and bar plots: HTML format
