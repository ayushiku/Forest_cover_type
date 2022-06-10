# -*- coding: utf-8 -*-
"""
Created on Sat May 21 12:38:52 2022

@author: Ayushi
"""

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score, precision_score,recall_score,roc_auc_score
data = pd.read_csv(r'C:\Users\Ayushi\Desktop\train.csv', index_col="Id")
data_test = pd.read_csv(r'C:\Users\Ayushi\Desktop/test.csv', index_col="Id")
ELU_CODE = {
    1:2702,2:2703,3:2704,4:2705,5:2706,6:2717,7:3501,8:3502,9:4201,
    10:4703,11:4704,12:4744,13:4758,14:5101,15:5151,16:6101,17:6102,
    18:6731,19:7101,20:7102,21:7103,22:7201,23:7202,24:7700,25:7701,
    26:7702,27:7709,28:7710,29:7745,30:7746,31:7755,32:7756,33:7757,
    34:7790,35:8703,36:8707,37:8708,38:8771,39:8772,40:8776
}

no_desc = [7,8,14,15,16,17,19,20,21,23,35]
stony = [6,12]
very_stony = [2,9,18,26]
extremely_stony = [1,22,24,25,27,28,29,30,31,32,33,34,36,37,38,39,40]
rubbly = [3,4,5,10,11,13]

surface_cover = {i:0 for i in no_desc}
surface_cover.update({i:1 for i in stony})
surface_cover.update({i:2 for i in very_stony})
surface_cover.update({i:3 for i in extremely_stony})
surface_cover.update({i:4 for i in rubbly})

soil_features = [f'Soil_Type{i}' for i in range(1,41)]
wilderness_features = [x for x in data.columns if x.startswith("Wilderness_Area")]

def r(x):
    if x+180>360:
        return x-180
    else:
        return x+180

def misc_features(data):
    df = data.copy()
    
    df["soil_type_count"] = df[soil_features].sum(axis=1)
    df["wilderness_area_count"] = df[wilderness_features].sum(axis=1)
    
    df['Soil_Type'] = 0
    for i in range(1,41):
        df['Soil_Type'] += i*df[f'Soil_Type{i}']
        
    df['Climatic_Zone'] = df['Soil_Type'].apply(
        lambda x: int(str(ELU_CODE[x])[0])
    )
    
    df['Geologic_Zone'] = df['Soil_Type'].apply(
        lambda x: int(str(ELU_CODE[x])[1])
    )
    
    df['Surface_Cover'] = df['Soil_Type'].apply(
        lambda x: surface_cover[x]
    )    

    
    
 
    df['Soil_12_32'] = df['Soil_Type32'] * df['Soil_Type12']
    df['Soil_Type23_22_32_33'] = df['Soil_Type23'] + df['Soil_Type22'] + df['Soil_Type32'] + df['Soil_Type33']
    
    
    df['Soil29_Area1'] = df['Soil_Type29'] + df['Wilderness_Area1']
    df['Soil3_Area4'] = df['Wilderness_Area4'] + df['Soil_Type3']
    
  
    df['Climate_Area2'] = df['Wilderness_Area2']*df['Climatic_Zone'] 
    df['Climate_Area4'] = df['Wilderness_Area4']*df['Climatic_Zone']
    df['Surface_Area1'] = df['Wilderness_Area1']*df['Surface_Cover'] 
    df['Surface_Area2'] = df['Wilderness_Area2']*df['Surface_Cover']   
    df['Surface_Area4'] = df['Wilderness_Area4']*df['Surface_Cover'] 
    
    
    
    
    for col, dtype in df.dtypes.iteritems():
        if dtype.name.startswith('float'):
            df[col] = df[col].astype('float64')
 
    df['Horizontal_Distance_To_Roadways_Log'] = [math.log(v+1) for v in df['Horizontal_Distance_To_Roadways']]
    df['Water Elevation'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
    df['Hydro_Fire_1'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']
    df['Hydro_Fire_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points'])
    df['Hydro_Road_1'] = abs(df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways'])
    df['Hydro_Road_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways'])
    df['Fire_Road_1'] = abs(df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Roadways'])
    df['Fire_Road_2'] = abs(df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways'])    
    df['EVDtH'] = df.Elevation - df.Vertical_Distance_To_Hydrology
    df['EHDtH'] = df.Elevation - df.Horizontal_Distance_To_Hydrology * 0.2
    df['Elev_3Horiz'] = df['Elevation'] + df['Horizontal_Distance_To_Roadways']  + df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Hydrology']
    df['Elev_Road_1'] = df['Elevation'] + df['Horizontal_Distance_To_Roadways']
    df['Elev_Road_2'] = df['Elevation'] - df['Horizontal_Distance_To_Roadways']
    df['Elev_Fire_1'] = df['Elevation'] + df['Horizontal_Distance_To_Fire_Points']
    df['Elev_Fire_2'] = df['Elevation'] - df['Horizontal_Distance_To_Fire_Points']
    
  
    df['EViElv'] = df['Vertical_Distance_To_Hydrology'] * df['Elevation']
    df['Aspect2'] = df.Aspect.map(r)

   
    df.fillna(0, inplace = True)
    
   
    for col, dtype in df.dtypes.iteritems():
        if dtype.name.startswith('int'):
            df[col] = pd.to_numeric(df[col], downcast ='integer')
        elif dtype.name.startswith('float'):
            df[col] = pd.to_numeric(df[col], downcast ='float')
            
    df.drop(columns = soil_features, inplace = True)
    df.drop(columns = ["Aspect"], inplace = True)
    df.drop(columns = ["Horizontal_Distance_To_Roadways"], inplace = True)
    
    return df
df_train = misc_features(data)
df_test = misc_features(data_test)
feature_cols = df_train.columns.to_list()
feature_cols.remove("Cover_Type")
def two_largest_indices(inlist):
    largest = 0
    second_largest = 0
    largest_index = 0
    second_largest_index = -1
    for i in range(len(inlist)):
        item = inlist[i]
        if item > largest:
            second_largest = largest
            second_largest_index = largest_index
            largest = item
            largest_index = i
        elif largest > item >= second_largest:
            second_largest = item
            second_largest_index = i        
    return largest_index, second_largest_index    



df_train_1_2 = df_train[(df_train['Cover_Type'] <= 2)]
df_train_3_4_6 = df_train[(df_train['Cover_Type'].isin([3,4,6]))]

X_train = df_train[feature_cols]
X_test = df_test[feature_cols]

X_train_1_2 = df_train_1_2[feature_cols]
X_train_3_4_6 = df_train_3_4_6[feature_cols]

y = df_train['Cover_Type']
y_1_2 = df_train_1_2['Cover_Type']
y_3_4_6 = df_train_3_4_6['Cover_Type']

test_ids = df_test.index

clf = ExtraTreesClassifier(n_estimators=500, random_state=42, max_depth=31, min_samples_split=2, criterion='entropy',
                          max_features=12, n_jobs=-1)
clf.fit(X_train, y)

clf_1_2 = ExtraTreesClassifier(n_estimators=500, random_state=42, max_depth=31, min_samples_split=2, criterion='gini',
                          max_features=12, n_jobs=-1)
clf_1_2.fit(X_train_1_2, y_1_2)

clf_3_4_6 = ExtraTreesClassifier(n_estimators=500, random_state=42, max_depth=31, min_samples_split=2, criterion='gini',
                          max_features=12, n_jobs=-1)
clf_3_4_6.fit(X_train_3_4_6, y_3_4_6)


vals_1_2 = {}
for e, val in enumerate(list(clf_1_2.predict_proba(X_test))):
    vals_1_2[e] = val


vals_3_4_6 = {}
for e, val in enumerate(list(clf_3_4_6.predict_proba(X_test))):
    vals_3_4_6[e] = val 


vals = {}
for e, val in enumerate(list(clf.predict(X_test))):
    vals[e] = val 
    

with open("submission.csv", "w") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(clf.predict_proba(X_test))):
        val[0] += vals_1_2[e][0]/1.3
        val[1] += vals_1_2[e][1]/1.1
        val[2] += vals_3_4_6[e][0]/3.4
        val[3] += vals_3_4_6[e][1]/4.0
        val[5] += vals_3_4_6[e][2]/3.6
        i,j = two_largest_indices(val)
        v = i  + 1
        outfile.write("%s,%s\n"%(test_ids[e],v))
actual=pd.read_csv(r'C:\Users\Ayushi\Desktop\traiin.csv', index_col="Id")
predicted=pd.read_csv(r'C:\Users\Ayushi\Desktop\python programs\submiision.csv', index_col="Id")
accuracy_check=accuracy_score(actual,predicted)
final_accuracy_to_be_printed=1-accuracy
df=pd.read_csv(r'C:\Users\Ayushi\Desktop\python programs\submission.csv')
df['Cover_Type'].hist()
plt.xlabel("Cover Type")
plt.ylabel("Number of trees of each cover type")
print ("Accuracy",final_accuracy_to_be_printed)