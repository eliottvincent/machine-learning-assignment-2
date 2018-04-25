import os
import csv

# Wilderness_Area (4 binary columns)
Wilderness_Area = {
	1: 'Rawah_Wilderness_Area',
	2: 'Neota_Wilderness_Area',
	3: 'Comanche_Peak_Wilderness_Area',
	4: 'Cache_la_Poudre_Wilderness_Area'
}

# Soil_Type (40 binary columns)
Soil_Type = {
	1: 'Cathedral_family',
	2: 'Vanet_Ratake_families',
	3: 'Haploborolis_family',
	4: 'Ratake_family',
	5: 'Vanet_family',
	6: 'Vanet_Wetmore_families',
	7: 'Gothic_family',
	8: 'Supervisor_family_Limber_families',
	9: 'Troutville_family',
	10: 'Bullwark_Catamount_families_Rock_outcrop_complex',
	11: 'Bullwark_Catamount_families_Rock_land_complex',
	12: 'Legault_family',
	13: 'Catamount_family_Rock_land',
	14: 'Pachic_Argiborolis',
	15: 'unspecified_family',
	16: 'Cryaquolis_family',
	17: 'Gateview_family',
	18: 'Rogert_family',
	19: 'Typic_Cryaquolis_Borohemists',
	20: 'Typic_Cryaquepts_Typic_Cryaquolls',
	21: 'Typic_Cryaquolls_Leighcan_family',
	22: 'Leighcan_family_till_substratum_extremely_bouldery',
	23: 'Leighcan_family_till_substratum_Typic_Cryaquolls_complex',
	24: 'Leighcan_family_extremely_stony',
	25: 'Leighcan_family_warm_extremely_stony',
	26: 'Granile_family_Catamount_families',
	27: 'Leighcan_family_warm_Rock_outcrop_complex',
	28: 'Leighcan_family_Rock_outcrop_complex',
	29: 'Como_Legault_families_complex',
	30: 'Como_family',
	31: 'Leighcan_family_Catamount_families_complex',
	32: 'Catamount_family_Rock_outcrop',
	33: 'Leighcan_family_Catamount families_Rock_outcrop_complex',
	34: 'Cryorthents_family',
	35: 'Cryumbrepts_family',
	36: 'Bross_family',
	37: 'Rock_outcrop-family',
	38: 'Leighcan_family_Moran_families',
	39: 'Moran_family_Cryorthents_Leighcan_family_complex',
	40: 'Moran_family_Cryorthents_Rock_land_complex'
}

# Cover_Type (7 types)
Cover_Type = {
	1: 'Spruce_Fir',
	2: 'Lodgepole_Pine',
	3: 'Ponderosa_Pine',
	4: 'Cottonwood_Willow',
	5: 'Aspen',
	6: 'Douglas_fir',
	7: 'Krummholz'
}

# Indexes
Wilderness_Area_start = 10
Wilderness_Area_end = 13
Soil_Type_start = 14
Soil_Type_end = 53
Cover_Type_start = 54

# Misc.
no_values = 0

def treatment():
	print('Treatment started')
	reader = loadCSV()
	writer = writeCSV()
	
	
	for row in reader:

		# Computing the values
		Wilderness_Area_value = Wilderness_Area_treatment(row)
		Soil_Type_value = Soil_Type_treatment(row)
		Cover_Type_value = Cover_Type_treatment(row)
		print(str(Wilderness_Area_value) + ', ' + str(Soil_Type_value) + ', ' + str(Cover_Type_value))

		# deleting rows
		del row[10:55]
		row.append(Wilderness_Area_value)
		row.append(Soil_Type_value)
		row.append(Cover_Type_value)
		writer.writerow(row)



	print('Treatment ended with ' + str(no_values) + ' errors')


def loadCSV():
	file = open('./data/DataSet.csv', 'r')
	return csv.reader(file, delimiter=',')

def writeCSV():
	file = open('./data/DataSet-cleaned.csv', 'w')
	return csv.writer(file, delimiter=',')

def Wilderness_Area_treatment(row):
	global no_values

	# Wilderness_Area_treatment_start
	# print('Wilderness_Area_treatment started')
	Wilderness_Area_value = 0
	x = 0
	for i in range(Wilderness_Area_start, Wilderness_Area_end+1):
		x += 1
		if row[i] == '1':
			Wilderness_Area_value = x
	# print('    Wilderness_Area is: ' + str(Wilderness_Area_value))
	if Wilderness_Area_value == 0:
		no_values += 1
	return Wilderness_Area_value
	# print('Wilderness_Area_treatment ended')
	# Wilderness_Area_treatment_end

def Soil_Type_treatment(row):
	global no_values

	# Soil_Type_treatment_start
	# print('Soil_Type_treatment started')
	Soil_Type_value = 0
	y = 0
	for j in range(Soil_Type_start, Soil_Type_end+1):
		y = y + 1
		if row[j] == '1':
			Soil_Type_value = y
	# print('    Soil_Type is: ' + str(Soil_Type_value))
	if Soil_Type_value == 0:
		no_values += 1
	return Soil_Type_value
	# print('Soil_Type_treatment ended')
	# Soil_Type_treatment_end

def Cover_Type_treatment(row):
	global no_values

	# Cover_Type_treatment_start
	# print('Cover_Type_treatment started')
	Cover_Type_value = 0
	k = Cover_Type_start
	Cover_Type_value = int(row[k])
	# print('    Cover_Type is: ' + str(Cover_Type_value))
	if Cover_Type_value == 0:
		no_values += 1
	return Cover_Type_value
	# print('Cover_Type_treatment ended')
	# Cover_Type_treatment_end
	
treatment()