# -*- coding: utf-8 -*-
"""
Model related functions defined on analysis 

To train the best model on new data, add the data to the raw csv data and use train_model function
Otherwise, use predict to predict new data

@author: icaromarley5
"""
from sklearn.tree import DecisionTreeClassifier
from joblib import dump,load
import pandas as pd

random_state = 100
model_name = 'model.joblib'
raw_data_path = 'KaggleV2-May-2016.csv'
target = 'No-show'

# builds a training dataset from raw data
# if a row is passed, it transforms the row for prediction
def build_training_data(one_row=None):   
    df = pd.read_csv(raw_data_path,parse_dates=['ScheduledDay','AppointmentDay'])
    # feature selection based on analysis
    columns = ['Scholarship', 'last_no_show', 'age_lesser_45', 'days_lesser_10']
    
    patient_id = None
    if one_row is not None:
        patient_id = one_row['PatientId']
        
    df = df.copy()
    # remove data that have ScheduledDay > AppointmentDay
    df['days'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    df = df[df['days'] >= 0]
    # change no show values to int
    df['No-show'].replace({'No':0,'Yes':1},inplace=True)
    # get last appointment
    df.sort_values('AppointmentDay',inplace=True)
    groupby_patient = df.sort_values('AppointmentDay').groupby('PatientId')
    df_last = groupby_patient.last()
    def compute_last_no_show(groupby):
        if groupby.shape[0] < 2:
            return 0
        return groupby['No-show'].shift(1).iloc[-1]
    df_last['last_no_show'] = groupby_patient\
        .apply(compute_last_no_show)    
    df = df_last.copy()
    # feature engineering based on analysis
    df['age_lesser_45'] = (df['Age'] <= 45).astype(int)
    df['days_lesser_10'] = (df['days'] <= 10).astype(int)
    
    result = df
    if patient_id:
        df.reset_index(inplace=True)
        result = df[df['PatientId'] == patient_id]
    result = result[columns + [target]]
    return result

# creates untrained best model based on analysis
def create_best_model(class_weight):
    return DecisionTreeClassifier(random_state=random_state,class_weight=class_weight)

def train_model():
    df = build_training_data()
    X = df[[column for column in df.columns if column != target]]
    y = df[target]
    class_weight  = dict(1 - y.value_counts()/y.shape[0])
    model = create_best_model(class_weight)
    model.fit(X,y)
    dump(model,model_name)
    return model

def load_model():
    return load(model_name)

def predict(data_row):
    row = build_training_data(data_row)
    if row.shape[0] > 0:
        row = row[[column for column in row.columns if column != target]]
        model = load_model()
        return model.predict(row)[0]
    return "invalid data. ScheduledDay > AppointmentDay"