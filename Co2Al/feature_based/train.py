from collections import Counter
from processing_data.get_data import get_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
import pickle
import joblib
import argparse
import json
from xgboost import plot_importance
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

warnings.filterwarnings("ignore")

def sort_by_another_list(df, column_name, sorter):
    sorterIndex = dict(zip(sorter, range(len(sorter))))
    df['Tm_Rank'] = df[column_name].map(sorterIndex)
    df.sort_values(['Tm_Rank'], ascending = [True], inplace = True)
    df.drop('Tm_Rank', 1, inplace = True)
    df.reset_index(inplace=True, drop=True)
    return df
def KQ(y_train, y_test,prediction_train,prediction_test):
    print('Accuracy_train:', accuracy_score(y_train, prediction_train))
    print('Accuracy_test:', accuracy_score(y_test, prediction_test))
    print('F1 score:', f1_score(y_test, prediction_test))
    print('Recall:', recall_score(y_test, prediction_test))
    print('Precision:', precision_score(y_test, prediction_test))
   # print('\n clasification report:\n', classification_report(y_test,prediction))
    print('\n confussion matrix:\n',confusion_matrix(y_test, prediction_test))
    
def main(args):
    number_day = str(args.number_day)
    # #Load data train ở dạng dataframe
    # f = open(args.load_data_train)
    # js = json.load(f)
    # data_his = pd.DataFrame(js)
    data_his = pd.read_csv("D:/AI Project/Icaller/data/data_nonreport2spam/data_after_clean/data_after_clean_" + number_day + "days/all_" + number_day + "days.csv")
    
    data_his.Spam = data_his.Spam.replace(2, 0)
    print(data_his.drop_duplicates(subset=['report']).Spam.value_counts())
    print(len(data_his))
    data_his = data_his.rename(columns={'ID': 'report', 'member_phone': 'phone_member'})
    data_his[['report', 'phone','phone_member']] = data_his[['report', 'phone', 'phone_member']].astype(str)
    data_train = get_data(data_his)
    data_train.to_csv("data_train_" + number_day + ".csv")
    print(data_train.Spam.value_counts())
    print(data_train.columns)
    data_train[['report']] = data_train[['report']].astype(str)
    X = data_train.drop(columns=['report', 'Spam'])
    Y = data_train[['report','Spam']]
    
    # Y = data_his.groupby(['report']).sum()
    # Y.reset_index(inplace=True)
    # Y = Y[['report','Spam']]
    # Y.loc[Y.Spam > 0, 'Spam'] = 1 
    # Y = Y[Y.report.isin(data_train.report)]
    # Y = sort_by_another_list(Y, 'report', data_train.report)
    del data_his, data_train
    print( len(X.columns) ,X.columns)
    # poly = PolynomialFeatures(2)
    # X = poly.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      Y['Spam'],
                                                      test_size=0.3,
                                                      random_state=3)
    XGB = xgb.XGBClassifier(objective='binary:logistic',
                            booster='gbtree',
                            colsample_bytree=1,
                            learning_rate=0.005,
                            max_depth=8,
                            min_child_weight=12,
                            scale_pos_weight=1,
                            n_estimators=3000,
                            random_state=1,
                            seed=1,
                            eta=0.3,
                            subsample=0.8,
                            tree_method='gpu_hist',
                            sampling_method='uniform')
    # define the datasets to evaluate each iteration
    evalset = [(X_train, y_train), (X_val, y_val)]
    # fit the model
    XGB.fit(X_train, y_train, eval_metric='auc', eval_set=evalset)
    # save the model to disk
    # joblib.dump(XGB, args.save_model)
    # print('Save to ', args.save_model)
    plot_importance(XGB)
    plt.show()
    prediction_train = XGB.predict(X_train)
    prediction_val =XGB.predict(X_val)
    KQ(y_train, y_val, prediction_train, prediction_val)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--load_data_train',
        help='load data train',
        default=
        "./data/data_after_clean/all_history_call_3month_for_report2spam.json")
    parser.add_argument('--save_model',
                        help='save model',
                        default='./checkpoints/XGB_nonreport2spam_180days.pkl')
    parser.add_argument(
                        '--number_day',
                        help='get number day',
                        default= 90)
    args = parser.parse_args()
    main(args)