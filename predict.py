import streamlit as st
import pickle
import numpy as np
import pandas as pd
# general libraries
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from tqdm.notebook import tqdm
from collections import Counter
warnings.filterwarnings("ignore")

# handling imbalance
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import make_pipeline as make_pipeline_imb

# visualization
import graphviz
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import export_text
from sklearn.tree import export_graphviz
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

# modelling
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.compose import make_column_selector, make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ensemble
import xgboost as xgb
from xgboost import XGBClassifier

# hypertuning
import mlflow
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

# #pycaret
# from pycaret.classification import *

#Metrics
from sklearn.metrics import f1_score, log_loss, roc_auc_score, precision_score, recall_score, matthews_corrcoef
from sklearn.feature_extraction import DictVectorizer

from PIL import Image

## Load Model

model = pickle.load(open('./xgb_final_1.pkl','rb'))


def predict_fraud(features):
    X=pd.DataFrame([features])
    prediction = model.predict(X)
    
    if prediction[0] == 1:
        message="Aww NOO! You're a fraud!"
        return message
    else:
        message="AWOO! You're good!"
        return message
    
    
def main():
        
    
    im = Image.open('./icon.png')
    st.set_page_config(page_title="Fraud Buster", page_icon = im, layout="centered")

    ##################
    job_category_dict={
        'Professionals':'Professionals',
        'Technicians and Associate Professionals':'TechniciansandAssociateProfessionals',
        'Managers':'Managers',
        'Craft and Related Trades Workers':'CraftandRelatedTradesWorkers',
        'Skilled Agricultural Forestry and Fisher Workers':'SkilledAgriculturalForestryandFisherWorkers',
        'Service and Sales Workers':'ServiceandSalesWorkers',
        'Plant and Machine Operators and Assemblers':'PlantandMachineOperatorsandAssemblers',
        'Elementary Occupations':'ElementaryOccupations',
    }
    job_category_options=[
                'Professionals', 
                'Technicians and Associate Professionals',
                'Managers', 
                'Craft and Related Trades Workers',
                'Skilled Agricultural Forestry and Fisher Workers',
                'Service and Sales Workers', 
                'Plant and Machine Operators and Assemblers',
                'Elementary Occupations'
                ]
    

    
    income_group_dict={
        'Low Income':'LowIncome', 
        'Lower Middle':'LowerMiddle',
        'Middle':'Middle', 
        'Upper Middle':'UpperMiddle',
    }
    income_group_options={
        'Low Income',
        'Lower Middle',
        'Middle',
        'Upper Middle',
    }
    
    category_dict={
                'Grocery Point-of-Sale': 'grocery_pos',
                'Gas Transport': 'gas_transport',
                'Health Fitness': 'health_fitness', 
                'Kids Pets': 'kids_pets',
                'Miscellaneous Net': 'misc_net', 
                'Travel': 'travel', 
                'Home': 'home', 
                'Entertainment': 'entertainment', 
                'Shopping Point-of-Sale': 'shopping_pos',
                'Grocery Net': 'grocery_net', 
                'Miscellaneous Point-of-Sale': 'misc_pos', 
                'Food Dining': 'food_dining', 
                'Personal Care': 'personal_care',
                'Shopping Net': 'shopping_net',
    }
    category_options=[
        'Grocery Point-of-Sale',
        'Gas Transport',
        'Health Fitness',
        'Kids Pets',
        'Miscellaneous Net',
        'Travel',
        'Home',
        'Entertainment',
        'Shopping Point-of-Sale',
        'Grocery Net',
        'Miscellaneous Point-of-Sale',
        'Personal Care',
        'Food Dining',
        'Shopping Net',
    ]
    
    trans_month_dict={
        'January':'Jan',
        'February':'Feb',
        'March':'Mar',
        'April':'Apr',
        'May':'May',
        'June':'Jun',
        'July':'Jul',
        'August':'Aug',
        'September':'Sep',
        'October':'Oct',
        'November':'Nov',
        'December':'Dec',
    }
    
    trans_month_options=[
        'January',
        'February',
        'March',
        'April',
        'May',
        'June',
        'July',
        'August',
        'September',
        'October',
        'November',
        'December',
    ]
    
    part_of_day_dict={
        'Early Morning':'early morning',
        'Breakfast':'breakfast',
        'Lunch':'lunch',
        'Afternoon':'afternoon',
        'Dinner':'dinner',
    }
    part_of_day_options=[
        'Early Morning',
        'Dinner',
        'Afternoon',
        'Lunch',
        'Breakfast',
    ]


    st.header("FRAUD BUSTER :female-detective:")
    st.markdown("Unmasking Deception, One Step Ahead!")
    # CUSTOMER DETAILS
    st.subheader(":dollar:  Transaction Details 	:dollar:", anchor=None, help=None, divider=False)
    amt=st.number_input("Amount of Transaction", 0, 20000)
    recency=st.number_input("How recent is the transaction?",0, 11214160)
    category= st.selectbox(
                "Type of Transaction",
                    options=category_options
                )
    trans_month= st.selectbox(
                "Month of Transaction",
                
                options=trans_month_options
                )
    part_of_day= st.selectbox(
                "Part-of-day of Transaction",
                options=part_of_day_options
                )
    freq12H_answer=st.selectbox(
                "Did the user transact 4 or more times in the span of 12 hours?",
                    options=['Yes',
                    'No']
                )
    t0_outlier_answer=st.selectbox(
                "Is the transaction tagged as an outlier?",
                    options=['Yes',
                    'No']
                )
    # CUSTOMER DETAILS
    st.subheader(":money_mouth_face: Customer Details :money_mouth_face:", anchor=None, help=None, divider=False)
    city_pop=st.number_input("Population of Customer's City", 0, 500000)
    distance=st.number_input("How far is the customer from the merchant?", 0, 155)
    freq_pop_ratio=st.number_input("What is the frequency of transaction to population ratio?", 0.0,1.0)
    km_per_trans=st.number_input("What is the distance travelled by customer between transactions? (km/transaction)", 0, 300)
    kmph=st.number_input("What is the rate of customer travel in between transactions? (kmph)",0, 795000)
    job_category= st.selectbox(
                    "Customer's Job Category",
                    options=job_category_options
    )
    income_group= st.selectbox(
                "Customer's Income Group",
                    options=income_group_options,
                )

    features={
    'city_pop':city_pop,
    'amt':amt,
    'recency':recency,
    'distance':distance,
    'freq_pop_ratio':freq_pop_ratio,
    'km_per_trans':km_per_trans,
    'kmph':kmph,
    'job_category': job_category_dict[job_category],
    'category': category_dict[category],
    'trans_month':trans_month_dict[trans_month],
    'part_of_day':part_of_day_dict[part_of_day],
    'income_group':income_group_dict[income_group],
    't0_outlier':[1 if t0_outlier_answer== 'Yes' else 0][0],
    'freq12H':[1 if freq12H_answer == 'Yes' else 0][0]
    }
    
    st.caption("This web app assumes that these features are readily present in your data.")

    if st.button(":ghost: Predict :ghost:"):
        
        result=predict_fraud(features)
        
        st.success(f'{result}')

if __name__=='__main__':
    main()
