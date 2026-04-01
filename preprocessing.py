# -*- coding: utf-8 -*-


# preprocessing.py
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import pickle
import joblib


model_dir = '/Users/gopal/Desktop/ML/Capstone_Project_Machine_Learning/models_over_sampling'
os.makedirs(model_dir,exist_ok=True)

def preprocess_and_split(df):
    # 📥 Load the dataset
    #df = pd.read_csv(path)
    print(f"✅ Loaded data with shape: {df.shape}")

    # Want to replace the space in column name
    new_col=[]
    for name in df.columns:
        new_name=name.replace(" ","_").replace("[","").replace("]","")
        new_col.append(new_name)
    #renaming
    df.rename(columns=dict(zip(df.columns,new_col)), inplace=True)
    print(df.head())

    # 🧹 Handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        df.dropna(inplace=True)
        print(f"🧹 Dropped rows with missing values: {missing_count}")
    else:
        print("✅ No missing values found")

    # 🧹 Remove duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df.drop_duplicates(inplace=True)
        print(f"🧹 Dropped duplicate rows: {duplicate_count}")
    else:
        print("✅ No duplicates found")



   
    X=df.drop(columns=['id', 'Product_ID', 'Type','Process_temperature_K','Machine_failure'])
    Y = df['Machine_failure']



    # 🌲 Feature Selection using Random Forest
    # 1. Initialize the Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42,class_weight='balanced')

	# 2. Wrap it in SelectFromModel
	# 'threshold' can be a number or a string like "mean" or "median"
    selector = SelectFromModel(estimator=rf, threshold=0.05) #Here we have use threshold value=0.05..it will be clear when we 
	#look at the feature importance graph

	# 3. Fit to data
    selector.fit(X,np.reshape(Y,(len(Y))))

	# Get the names of the features that were kept
    important_feature_names = selector.get_feature_names_out()
    
    selected_features=os.path.join(model_dir,'selected_features.txt')

    with open(selected_features, 'w') as f:
        for feat in important_feature_names:
            f.write(f"{feat}\n")
            
        #Transform into selected features
    X_transform=selector.transform(X)
    selector_path=os.path.join(model_dir,"selector.pkl")
    joblib.dump(selector, selector_path)

    # ✂️ Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_transform,Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
        )

    return X_train, X_test, y_train, y_test, important_feature_names, selector_path