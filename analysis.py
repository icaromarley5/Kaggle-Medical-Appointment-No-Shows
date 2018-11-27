# -*- coding: utf-8 -*-
"""
Script for anaylsis  and defining pipeline
For usable functions please see model_tools module

Data origin: https://www.kaggle.com/joniarroba/noshowappointments/home
The data is about medical apointments in Vitória - ES - Brazil.
The objective of this work is to predict if a patient will show up on a apointment.

@author: icaromarley5
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000

file_path = 'KaggleV2-May-2016.csv'

df = pd.read_csv(file_path,parse_dates=['ScheduledDay','AppointmentDay'])

random_state = 0
target = 'No-show'
columns = [column for column in df.columns if column != target] # this variable is a reminder of the selected columns for prediction
'''
From data description:
PatientId - Identification of a patient 
AppointmentID - Identification of each appointment 
Gender = Male or Female . Female is the greater proportion, woman takes way more care of they health in comparison to man. 
AppointmentDay = The day of the actuall appointment,  when they have to visit the doctor. 
ScheduledDay = The day someone called or registered the appointment, this is before appointment of course. 
Age = How old is the patient. 
Neighbourhood = Where the appointment takes place. 
Scholarship = Ture of False . Observation, this is a broad topic, consider reading this article https://en.wikipedia.org/wiki/Bolsa_Fam%C3%ADlia 
Hipertension = True or False 
Diabetes = True or False 
Alcoholism = True or False 
Handcap = True or False 
SMS_received = 1 or more messages sent to the patient. 
No-show = True or False.
'''


'''
basic checking to see if task is possible
.independent rows assumption
.nulls on dependant variable
.proportion on dependant variable

basic data checking
.type casting
.typos on values
.invalid values (out of possible range, so on)
.inconsistencies between columns
'''
df.isnull().any() # no nulls

df['AppointmentDay'].dt.hour.unique() # doesn't have hour values
df['days'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df[df['days'] < 0].shape[0]/ df.shape[0]
# 33 % are bellow 0
columns.append('days')

df['Age'].max() # ok
df['Age'].min() # -1, data creator says is for unborn babies

df['PatientId'].unique().shape[0] == df.shape[0]
# patients are not unique!
# to not break uncorrelated rows assumption, I'll groupby for patients and get the last appointment
# since appointment will be a imprtant variable, 
#I'll drop de 33% of the data where Appointment date is ealrier than Schedule datenão
df = df[df['days']>= 0]
df = df.sort_values('AppointmentDay')
df['No-show'].unique()
df['AppointmentDay'].dt.year.unique() 
# since the appointments happened in less than a year, most variables are still the same (exeample: Neighbourhood)
# so there is no extra information that is needed to be computed on grouping the data by patient
df['No-show'].replace({'No':0,'Yes':1},inplace=True) # type parsing
groupby_patient = df.sort_values('AppointmentDay').groupby('PatientId')
df_last = groupby_patient.last()
# although, some features might be engineering at this point, like
# if the patient is used to receive sms warnigns but not received one on his/her last appointment
def compute_sms(groupby):
    if groupby.shape[0] < 2:
        return 0
    if (groupby.iloc[-2]['SMS_received']) == 1 and\
        (groupby.iloc[-1]['SMS_received'] == 0):
        return 1
    return 0
df_last['no_last_sms'] = groupby_patient\
    .apply(compute_sms)
# percent of appointments that a patient missed
def compute_percent_no_show(groupby):
    if groupby.shape[0] < 2:
        return 0
    total_counts = groupby.shape[0] - 1
    return groupby['No-show'].iloc[:-1].sum()/ total_counts
df_last['percent_no_show'] = groupby_patient\
    .apply(compute_percent_no_show)
# if patient missed his/her last appointment
def compute_last_no_show(groupby):
    if groupby.shape[0] < 2:
        return 0
    return groupby['No-show'].shift(1).iloc[-1]
df_last['last_no_show'] = groupby_patient\
    .apply(compute_last_no_show)

df = df_last.copy()
del df_last
columns.remove('PatientId')
columns += ['percent_no_show','last_no_show','no_last_sms']

df[target].value_counts() # valid proportions
# dummification of gender
df['Gender'].unique()
df['gender_m'] = (df['Gender'] == 'M').astype(int) # 
columns.remove('Gender')
columns.append('gender_m')
# other data checks
df['SMS_received'].unique()
df['Hipertension'].unique()
df['Diabetes'].unique()
df['Alcoholism'].unique()
df['Scholarship'].unique()
df['Handcap'].unique()
# variaveis numéricas
df['Neighbourhood'].unique()
df['Neighbourhood'].unique().shape 
# 80 values about the regions of Vitória
# this row can be transformed to hold the values of the administrative areas
# all admnistrative areas are present in this link https://pt.wikipedia.org/wiki/Jardim_Camburi
dict_regions = {
 'AEROPORTO': 'GOIABEIRAS',
 'ANDORINHAS': 'MARUÍPE',
 'ANTÔNIO HONÓRIO': 'GOIABEIRAS',
 'ARIOVALDO FAVALESSA': 'SANTO ANTÔNIO',
 'BARRO VERMELHO': 'PRAIA DO CANTO',
 'BELA VISTA': 'SANTO ANTÔNIO',
 'BENTO FERREIRA': 'JUCUTUQUARA',
 'BOA VISTA': 'JARDIM DA PENHA',
 'BONFIM': 'MARUÍPE',
 'CARATOÍRA': 'SANTO ANTÔNIO',
 'CENTRO': 'CENTRO',
 'COMDUSA': 'SÃO PEDRO',
 'CONQUISTA': 'SÃO PEDRO',
 'CONSOLAÇÃO': 'JUCUTUQUARA',
 'CRUZAMENTO': 'JUCUTUQUARA',
 'DA PENHA': 'MARUÍPE',
 'DE LOURDES': 'JUCUTUQUARA',
 'DO CABRAL': 'SANTO ANTÔNIO',
 'DO MOSCOSO': 'CENTRO',
 'DO QUADRO': 'SANTO ANTÔNIO',
 'ENSEADA DO SUÁ': 'PRAIA DO CANTO',
 'ESTRELINHA': 'SANTO ANTÔNIO',
 'FONTE GRANDE': 'CENTRO',
 'FORTE SÃO JOÃO': 'JUCUTUQUARA',
 'FRADINHOS': 'JUCUTUQUARA',
 'GOIABEIRAS': 'GOIABEIRAS',
 'GRANDE VITÓRIA': 'SANTO ANTÔNIO',
 'GURIGICA': 'JUCUTUQUARA',
 'HORTO': 'JUCUTUQUARA',
 'ILHA DAS CAIEIRAS':'SÃO PEDRO',
 'ILHA DAS CAIERAS': 'SÃO PEDRO', # typo
 'ILHA DE SANTA MARIA': 'JUCUTUQUARA',
 'ILHA DO BOI': 'PRAIA DO CANTO',
 'ILHA DO FRADE': 'PRAIA DO CANTO',
 'ILHA DO PRÍNCIPE': 'CENTRO',
 'INHANGUETÁ': 'SANTO ANTÔNIO',
 'ITARARÉ': 'MARUÍPE',
 'JABOUR': 'GOIABEIRAS',
 'JARDIM CAMBURI': 'JARDIM CAMBURI',
 'JARDIM DA PENHA': 'JARDIM DA PENHA',
 'JESUS DE NAZARÉ': 'JUCUTUQUARA', # typo
 'JESUS DE NAZARETH':'JUCUTUQUARA',
 'JOANA D´ARC': 'MARUÍPE',
 'JUCUTUQUARA': 'JUCUTUQUARA',
 'MARIA ORTIZ': 'GOIABEIRAS',
 'MARUÍPE': 'MARUÍPE',
 'MATA DA PRAIA': 'JARDIM DA PENHA',
 'MONTE BELO': 'JUCUTUQUARA',
 'MORADA DE CAMBURI': 'JARDIM DA PENHA',
 'MÁRIO CYPRESTE': 'SANTO ANTÔNIO',
 'NAZARÉ': 'JUCUTUQUARA',
 'NOVA PALESTINA': 'SÃO PEDRO',
 'PARQUE INDUSTRIAL': 'JARDIM CAMBURI',
 'PARQUE MOSCOSO': 'CENTRO',
 'PIEDADE': 'CENTRO',
 'PONTAL DE CAMBURI': 'JARDIM DA PENHA',
 'PRAIA DO CANTO': 'PRAIA DO CANTO',
 'PRAIA DO SUÁ': 'PRAIA DO CANTO',
 'REDENÇÃO': 'SÃO PEDRO',
 'REPÚBLICA': 'JARDIM DA PENHA',
 'RESISTÊNCIA': 'SÃO PEDRO',
 'ROMÃO': 'JUCUTUQUARA',
 'SANTA CECÍLIA': 'MARUÍPE',
 'SANTA CLARA': 'CENTRO',
 'SANTA HELENA': 'PRAIA DO CANTO',
 'SANTA LUÍZA': 'PRAIA DO CANTO',
 'SANTA LÚCIA': 'PRAIA DO CANTO',
 'SANTA MARTHA': 'MARUÍPE',
 'SANTA TEREZA': 'SANTO ANTÔNIO',
 'SANTO ANDRÉ': 'SÃO PEDRO',
 'SANTO ANTÔNIO': 'SANTO ANTÔNIO',
 'SANTOS DUMONT': 'MARUÍPE',
 'SANTOS REIS': 'SÃO PEDRO',
 'SEGURANÇA DO LAR': 'GOIABEIRAS',
 'SOLON BORGES': 'GOIABEIRAS',
 'SÃO BENEDITO': 'MARUÍPE',
 'SÃO CRISTÓVÃO': 'MARUÍPE',
 'SÃO JOSÉ': 'SÃO PEDRO',
 'SÃO PEDRO': 'SÃO PEDRO',
 'TABUAZEIRO': 'MARUÍPE',
 'UNIVERSITÁRIO': 'SANTO ANTÔNIO',
 'NAZARETH':'JUCUTUQUARA',
 'VILA RUBIM': 'CENTRO'}
df['Neighbourhood'] = df['Neighbourhood'].replace(dict_regions)
df['Neighbourhood'].unique() # only 10 values now

# discard AppointmentID
columns.remove('AppointmentID')

'''
Train test split.
Stratified split to mantain whatever porpotions the target variable have
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[columns],df[target],test_size=.3,random_state=random_state,shuffle=True,stratify = df[target])
y_train.value_counts()/y_train.shape[0]
y_test.value_counts()/y_test.shape[0]
del df
X_train[target] = y_train
X_test[target] = y_test
del y_train,y_test
# keep train and test together to train last model
X = pd.concat([X_train,X_test],axis=0)

'''
EDA on training data
.feature visualization
.feature transformations
.feature engineering
.feature selection
'''
# univariate anaylisis of categorical features
categorical_columns = ['gender_m','Neighbourhood',
 'Scholarship','Hipertension','Diabetes','Alcoholism',
 'SMS_received','Handcap','last_no_show','no_last_sms']

for column in categorical_columns:
    X_train[column].value_counts().plot(kind='bar')
    plt.title(column)
    plt.show()
'''
more women than men
MARUÍPE holds more appointments
more without government aid (scholarship)
more healthy 
few alcoholics
sms is more balanced
almost no one with some kind of handcap
few patients missed last appointment
most received a sms or never received one
'''

# dummification on Neighbourhood
neighbourhood_df = pd.get_dummies(X_train['Neighbourhood'])
columns.remove('Neighbourhood')
columns += list(neighbourhood_df.columns)
X_train = pd.concat([X_train,neighbourhood_df],axis=1)
# since ILHAS OCEÂNICAS DE TRINDADE have almost no data, this feature can be discarded
columns.remove( 'ILHAS OCEÂNICAS DE TRINDADE')
# since handcap have few more than 0 values, this featuer can be binarized
X_train['Handcap'] = (X_train['Handcap'] < 1).astype(int)
column = 'Handcap'
X_train[column].value_counts().plot(kind='bar') # still unbalanced
plt.title(column)
plt.show()
# remove unbalanced columns
columns.remove('Handcap')
columns.remove('Alcoholism')

# univariate analysis on numerical features
numerical_columns = [
 'Age',
 'days',
 'percent_no_show',
]

for column in numerical_columns:
    X_train[column].plot(kind='hist')
    plt.title(column)
    plt.show()
'''
most have age < 60, with many children and newborns
msot patients waited less than a month 
most didn't miss or did't have appointments last appointments
'''
(X_train['percent_no_show'] == 1).value_counts()
# percent no show is very unbalanced, binarization could help
X_train['always_no_show'] = (X_train['percent_no_show'] == 1).astype(int)
columns.remove('percent_no_show')
columns.append('always_no_show')
column = 'always_no_show'
X_train[column].value_counts().plot(kind='bar') 
plt.title(column)
plt.show()

'''
feature engineering no training set
'''
# time features on appointment and schedule date
X_train['appointment_hour'] = X_train['AppointmentDay'].dt.hour
X_train['appointment_day'] = X_train['AppointmentDay'].dt.day
X_train['appointment_month'] = X_train['AppointmentDay'].dt.month
X_train['appointment_year'] = X_train['AppointmentDay'].dt.year
X_train['scheduled_hour'] = X_train['ScheduledDay'].dt.hour
X_train['scheduled_day'] = X_train['ScheduledDay'].dt.day
X_train['scheduled_month'] = X_train['ScheduledDay'].dt.month
X_train['scheduled_year'] = X_train['ScheduledDay'].dt.year

time_columns = ['appointment_hour','appointment_day','appointment_month','appointment_year',
            'scheduled_hour','scheduled_day','scheduled_month','scheduled_year']

columns = columns + time_columns

for column in time_columns:
    X_train[column].plot(kind='hist')
    plt.title(column)
    plt.show()

'''
appointment:
.there is no hour values
.most days are in the beginning of the month
. most happened in may
. only in 2016
schedule:
. hour is right skewed
.most days are in the beginning, middle and end of the month
.most happened on the first half of the first semester
.most heppened in 2016
'''
# remove features with low variance
columns.remove('scheduled_year')
columns.remove('appointment_year')
columns.remove('appointment_hour')
columns.remove('AppointmentDay')
columns.remove('ScheduledDay')

# bivariate analysis on categorical features

import scipy.stats as st

categorical_columns = [
 'CENTRO',
 'GOIABEIRAS',
 'JARDIM CAMBURI',
 'JARDIM DA PENHA',
 'JUCUTUQUARA',
 'MARUÍPE',
 'PRAIA DO CANTO',
 'SANTO ANTÔNIO',
 'SÃO PEDRO',
 'Scholarship',
 'Hipertension',
 'Diabetes',
 'SMS_received',
 'last_no_show',
 'no_last_sms',
 'gender_m',
 'always_no_show',]

def plot_categorical(columns,target=target):
    for column in columns:
        true_counts = X_train.groupby(column)[target].value_counts().unstack()
        counts =  true_counts.divide(true_counts.sum(axis=1),axis=0)
        chi2 = st.chi2_contingency(true_counts)[1]
        ax = counts.plot(kind='bar',stacked=True)
        rects = ax.patches
        labels = true_counts.values.flatten('F')
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y()+.4*height, label,
                    ha='center', va='bottom')
        plt.title('Chi2: {}'.format(chi2))
        plt.show()      
plot_categorical(categorical_columns)
'''
Chi2 P-value less than 0.05: CENTRO, JARDIM CAMBURI, PRAIA DO CANTO, SANTO ANTÔNIO, SÃO PEDRO, no_last_sms, gender_m
In percentages:
less patients missed appointments on GOIBABEIRAS, JARDIM DA PENHA
more patients missed appointments on JUCUTUQUARA, MARUÍPE
more patients that recieved government aid missed appointments
less patients with hipertension missed appointments
less patients with diabetes missed appointments
less patients that received sms missed appointments
more patients that missed last appointment missed a new one
more patients that missed all appointments missed a new one
'''
'''
discarding features with p-value higher than 0.05
'''
remove_columns = ['CENTRO','JARDIM CAMBURI','PRAIA DO CANTO','SANTO ANTÔNIO','SÃO PEDRO','no_last_sms','gender_m']
columns = [column for column in columns if column not in remove_columns]

# bivariate analysis on numerical features
numerical_columns = [
'scheduled_hour',
 'Age',
 'days',
 'appointment_day',
 'appointment_month',
 'scheduled_day',
 'scheduled_month']

X_train['all'] = ''
for column in numerical_columns:
    ax = sns.violinplot(y=column,x='all',hue=target,data=X_train,legend=['0','1'],split=True,scale='area')
    result = st.pointbiserialr(X_train[target],X_train[column])
    plt.title('pointb corr {:.2f} p-value {:.2f}'.format(result.correlation,result.pvalue))
    ax.axes.get_xaxis().set_visible(False)
    plt.show()

'''
Considering Point-Biserial Correlation, almost all features have weak or no linear relationship
less patients with a scheduled hour before 8h, missed appointments
more patients with age up to 45 missed appointments
less patients with age between 45 and 80 missed appointments
less patients with wait time less than 10 days missed appointments
appointment day and month have no pattern
scheduled day have no pattern 
less patients with scheduled month around may and june missed appointments
'''
# feature binarization to extract found patterns
X_train['scheduled_hour_8'] = (X_train['scheduled_hour'] <= 8).astype(int)
X_train['age_lesser_45'] = (X_train['Age'] <= 45).astype(int)
X_train['age_45_to_80'] = ((45 < X_train['Age']) & (X_train['Age'] <= 80)).astype(int)
X_train['days_lesser_10'] = (X_train['days'] <= 10).astype(int)
X_train['scheduled_month_greater_5'] = (X_train['scheduled_month'] > 5).astype(int)
new_columns = ['scheduled_hour_8','age_lesser_45','age_45_to_80','days_lesser_10','scheduled_month_greater_5'] 
columns = columns + new_columns 

plot_categorical(new_columns)
'''
All chi2 p-values bellow 0.05
'''

# discarding binarized features
remove_columns =  [
 'scheduled_hour',
 'Age',
 'days',
 'appointment_day',
 'appointment_month',
 'scheduled_day',
 'scheduled_month']
columns = [column for column in columns if column not in remove_columns]

'''
feature selection
'''
# using dython module, from https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
from dython.nominal import associations
associations_results = associations(X_train[columns+[target]],nominal_columns=columns+[target],return_results=True,plot=False)

associations_results[target].sort_values()
'''
all features have weak or no association with the dependant variable
discarding features with low association from Cramer's V (<0.05)
'''
remove_columns = [
    'GOIABEIRAS',
    'Diabetes',
    'JUCUTUQUARA',
    'MARUÍPE',
    'SMS_received',
    'JARDIM DA PENHA',
    'scheduled_month_greater_5',
    'scheduled_hour_8'
]
plot_categorical(remove_columns)
columns = [column for column in columns if column not in remove_columns]

'''
forward feature selection
'''
# first feature: age_lesser_45 (best association)
sns.barplot(x='age_lesser_45',y=target,data=X_train)
plt.show()
# age_45_to_80
sns.barplot(x='age_lesser_45',y=target,hue='age_45_to_80',data=X_train)
plt.show()
associations_results['age_45_to_80'].sort_values()
'''
the adition of age_45_to_80 disturbs the association between age_lesser_45 and dependant variable
discard feature
'''
columns.remove('age_45_to_80')
associations_results = associations(X_train[columns+[target]],nominal_columns=columns+[target],return_results=True,plot=False)
# last_no_show
sns.barplot(x='age_lesser_45',y=target,hue='last_no_show',data=X_train)
plt.show()
print(associations_results['last_no_show'].sort_values())
'''
when last_no_show = 1, the rate of missed appointments is higher
those features have no association
'''
# always_no_show
sns.factorplot("last_no_show", target, hue='age_lesser_45',col="always_no_show", data=X_train, kind="bar")
plt.show()
associations_results['always_no_show'].sort_values()
'''
always_no_show and last_no_show are strongly associated
considering last_no_show, always_no_show almoest doesn't affect the target variable
discard feature
'''
columns.remove('always_no_show')

associations_results = associations(X_train[columns+[target]],nominal_columns=columns+[target],return_results=True,plot=False)
# Hipertension
sns.factorplot("last_no_show",target, hue='age_lesser_45',col='Hipertension', data=X_train, kind="bar")
plt.show()
associations_results['Hipertension'].sort_values()
'''
hipertension disturbs the associations
discard feature
'''
columns.remove('Hipertension')

associations_results = associations(X_train[columns+[target]],nominal_columns=columns+[target],return_results=True,plot=False)
# days_lesser_10
sns.factorplot("last_no_show",target, hue='age_lesser_45',col='days_lesser_10', data=X_train, kind="bar")
plt.show()
associations_results['days_lesser_10'].sort_values()
'''
days_lesser_10 have no association with other features
when days_lesser_10 = 1 the rate of missed appointments is lower
'''
# Scholarship
sns.factorplot("last_no_show",target, hue='age_lesser_45',row='days_lesser_10',col='Scholarship', data=X_train, kind="bar")
plt.show()
associations_results['Scholarship'].sort_values()
'''
government aid is weakly associated with age_lesser_45
when days_lesser_10 =1 last_no_show = 1 scholarship=0, the rate of missed appointments are lower
when days_lesser_10=1 last_no_show=0 scholarship=1,  the rate of missed appointments are lower
'''

'''
model selection
'''

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,matthews_corrcoef,accuracy_score
from sklearn.dummy import DummyClassifier

y_true = X_train[target]

class_weight  = dict(1 - X_train[target].value_counts()/X_train.shape[0])

model = LogisticRegression(random_state=random_state,class_weight=class_weight)
model.fit(X_train[columns],X_train[target])
print(str(model).split('(')[0])
y_pred = model.predict(X_train[columns])
print(confusion_matrix(y_true,y_pred))
print("MCC",matthews_corrcoef(y_true,y_pred))
print('ACC',accuracy_score(y_true,y_pred))

model = DecisionTreeClassifier(random_state=random_state,class_weight=class_weight)
model.fit(X_train[columns],X_train[target])
print(str(model).split('(')[0])
y_pred = model.predict(X_train[columns])
print(confusion_matrix(y_true,y_pred))
print("MCC",matthews_corrcoef(y_true,y_pred))
print('ACC',accuracy_score(y_true,y_pred))

print('dummy random (coin flip)')
dummy = DummyClassifier('uniform',random_state=random_state)
dummy.fit(X_train[columns],X_train[target])
y_pred = dummy.predict(X_train[columns])
print(confusion_matrix(y_true,y_pred))
print("MCC",matthews_corrcoef(y_true,y_pred))
print('ACC',accuracy_score(y_true,y_pred))

print('dummy stratified')
dummy = DummyClassifier('stratified',random_state=random_state)
dummy.fit(X_train[columns],X_train[target])
y_pred = dummy.predict(X_train[columns])
print(confusion_matrix(y_true,y_pred))
print("MCC",matthews_corrcoef(y_true,y_pred))
print('ACC',accuracy_score(y_true,y_pred))

'''
Decision Tree had the best scores, with MCC and ACC above the dummy models
ACC: 23.30% higher than coin flip model and 3.8% higher than stratified model
Both MCC of the dummies are next to zero (indicates a randomic model, which is true)
'''
# Tree analysis
importances = model.feature_importances_
indices = np.argsort(importances)
plt.figure()
plt.title("Feature importances")
plt.barh(range(X_train[columns].shape[1]), importances[indices],
       color="r", align="center")
plt.yticks(range(X_train[columns].shape[1]), X_train[columns].columns[indices])
plt.show()

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydot

dot_data = StringIO()
export_graphviz(model, out_file=dot_data,feature_names=columns, 
                filled=True, rounded=True,class_names=['False','True'],
                precision=2,proportion=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
'''
age_lesser_45 is the most important feature, last_no_show is the second 
scholarship had almost no importance
most of the proportions of the classes on the leaf nodes are close
this indicates that the features aren't enough to define the classification boundary
age_lesser_45 splits the tree on nodes of non missed appointment (when the feature=0)
and nodes of missed appointment (=1)
inside age_lesser_45=0, there are a few missed appointment predominand nodes under last_no_show = 1
under age_lesser_45 = 1, almost all nodes are missed appointment predominant
'''

'''
Conclusion 1: the fitting process could be better with more relevant features
'''

'''
test on the test set
'''
def build_dataset(df,test=False):
    X = df.copy()
    target = 'No-show'
    columns = ['Scholarship', 'last_no_show', 'age_lesser_45', 'days_lesser_10']
    X['age_lesser_45'] = (X['Age'] <= 45).astype(int)
    X['days_lesser_10'] = (X['days'] <= 10).astype(int)
    if not test:       
        return X[columns],X[target],columns
    return X[columns],columns

X__,columns = build_dataset(X_test,test=True)

y_true = X_test[target]
print(str(model).split('(')[0])
y_pred = model.predict(X__)
print(confusion_matrix(y_true,y_pred))
print("MCC",matthews_corrcoef(y_true,y_pred))
print('ACC',accuracy_score(y_true,y_pred))

print('dummy random')
dummy = DummyClassifier('uniform',random_state=random_state)
dummy.fit(X_train[columns],X_train[target])
y_pred = dummy.predict(X__)
print(confusion_matrix(y_true,y_pred))
print("MCC",matthews_corrcoef(y_true,y_pred))
print('ACC',accuracy_score(y_true,y_pred))

print('dummy stratified')
dummy = DummyClassifier('stratified',random_state=random_state)
dummy.fit(X_train[columns],X_train[target])
y_pred = dummy.predict(X__)
print(confusion_matrix(y_true,y_pred))
print("MCC",matthews_corrcoef(y_true,y_pred))
print('ACC',accuracy_score(y_true,y_pred))

'''
ACC and MCC of the Decision Tree model fell by 0.9% and by 8.16%, respectively 
those values are low and espected, since its unseen data
although the ACC and MCC fell a little, the situation on the training set is still the same, with
ACC 24.02% higher than the coin flip model and 3.45% higher than the stratified model
MCC of the dummies are still next do zero
'''

'''
Conclusion 2: This model can be used to predict the dependant variable better than the dummies models and can be used in such task
'''
'''
Training final model
'''

X_all,y_all,columns = build_dataset(X)
class_weight  = dict(1 - y_all.value_counts()/y_all.shape[0])
model = DecisionTreeClassifier(random_state=random_state,class_weight=class_weight)
model.fit(X_all,y_all)
# tree analysis 
importances = model.feature_importances_
indices = np.argsort(importances)
plt.figure()
plt.title("Feature importances")
plt.barh(range(X_all[columns].shape[1]), importances[indices],
       color="r", align="center")
plt.yticks(range(X_all[columns].shape[1]), X_all[columns].columns[indices])
plt.show()

dot_data = StringIO()
export_graphviz(model, out_file=dot_data,feature_names=columns, 
                filled=True, rounded=True,class_names=['False','True'],
                precision=2,proportion=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
'''
Similar tree and importances
'''

'''
Summary of this analysis: 
.the features present on the data had low association with the dependant variable
.the features extracted had low association with the dependant variable
.a model was built with the extracted features
.the model built performed better than coin flip guessing and proportion guessing
.the performance of the model is almost the same with unseen data
'''