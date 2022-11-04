from pyexpat import model
import xgboost as xgb
from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from processing_data.get_data import get_data
from sklearn.preprocessing import PolynomialFeatures
from utils import *
import pandas as pd
import numpy as np
import joblib
import json
import argparse
import warnings
import random
warnings.filterwarnings("ignore")

def sort_by_another_list(df, column_name, sorter):
    sorterIndex = dict(zip(sorter, range(len(sorter))))
    df['Tm_Rank'] = df[column_name].map(sorterIndex)
    df.sort_values(['Tm_Rank'], ascending=[True], inplace=True)
    df.drop('Tm_Rank', 1, inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

def main(args):
    #Load data test 
    # f = open(args.load_data_test)
    # js = json.load(f)
    # data_his = pd.DataFrame(js)
    data = pd.read_csv("D:\AI Project\Icaller_spam_xgb\data\data_for_test\data_test_90days.csv")
    data[['phone','phone_member']] = data[['phone', 'phone_member']].astype(str)
    get_nonspam = data.loc[data.Spam == 2].phone.unique()
    print(len(get_nonspam))
    get_nonspam = get_nonspam[:70000]
    # get_nonspam = random.choices(get_nonspam, k = num_nonphone)
    print(len(get_nonspam))
    
    data_his = data[~data.phone.isin(get_nonspam)]
    print(len(data_his.loc[data_his.Spam == 2].phone.unique()))
    data_his.reset_index(inplace=True, drop=True)
    
    Y = data_his.drop_duplicates(subset=['report'])
    Y.reset_index(inplace=True)
    Y = Y[['report','Spam']]
    Y.loc[Y.Spam == 2, 'Spam']=0
    print(Y.Spam.value_counts())
    
    df_test = get_data(data_his)
    
    Y = Y[Y.report.isin(df_test.report)]
    Y = sort_by_another_list(Y, 'report', df_test.report)
    X = df_test.drop(columns=['report'])
    print( len(X.columns) ,X.columns)
    del data_his, df_test
    poly = PolynomialFeatures(2)
    X = poly.fit_transform(X)
    #Load model
    model = joblib.load(args.load_model)
    
    # Kết quả
    prediction_test =model.predict(X)
    proba_test =model.predict_proba(X)
    pd.DataFrame(data={'predict': prediction_test, 'proba_0': proba_test[:,0], 'proba_1': proba_test[:,1]}).to_csv("result.csv")
    proba_test = proba_test[:,1]
    prediction_test= np.where(proba_test > 0.9, 1, 0)
    print(accuracy_score(Y['Spam'],prediction_test))
    print(classification_report (Y['Spam'], prediction_test))
    print(confusion_matrix(Y['Spam'], prediction_test))
    print("AUC: ", roc_auc_score(Y['Spam'], proba_test))

if __name__ == '__main__':  

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_data_test', help='load data train', default="./data/data_for_test/data.json")
    parser.add_argument('--load_model', help='load model', default='./checkpoints/XGB_nonreport2spam_90days.pkl')
    args = parser.parse_args() 
    main(args)