from processing_data.get_data import get_data
from processing_data.processing_function import remove_prefix
from sklearn.preprocessing import PolynomialFeatures
import argparse
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
import json
warnings.filterwarnings("ignore")

def main(args):
    #Load data history của 1 SĐT 
    f = open(args.load_data)
    js = json.load(f)
    data_his = pd.DataFrame(js)
    data_his = data_his.astype({
                                'phone': 'str',
                                'time': 'str',
                                'phone_member': 'str'
                                })
    data_his['phone_remove'] = data_his.phone.apply(remove_prefix)
    df_phone = data_his[['phone','phone_remove']].drop_duplicates(subset=['phone_remove'])
    #feature engineering
    df_pool = get_data(data_his)
    
    df_pool = df_pool.replace(df_phone.phone_remove.unique(), df_phone.phone.unique())
    del data_his,df_phone
    X = df_pool.drop(columns=['phone', 'Spam', 'time_join'])
    poly = PolynomialFeatures(2)
    X = poly.fit_transform(X)
    #load model ở checkpoints
    XGB = xgb.XGBClassifier()
    XGB.load_model(fname=args.load_model)
    #Kết quả dự đoán
    y_pred = XGB.predict(X)
    y_prob = XGB.predict_proba(X)
    output = [{'phone': phone,  
              'Spam': pred,  
              'Score': score} for phone, pred, score in zip(df_pool.phone.tolist(), y_pred.tolist(),y_prob.max(axis=1).tolist())]
    #Save output to json
    with open(args.return_output, "w+") as outfile:
        json.dump(output, outfile)
if __name__ == '__main__':  

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_data', help='load data history', default="./data/data_for_predict/his_3month_for_predict.json")
    parser.add_argument('--load_model', help='load model', default='./checkpoints/model_xgb.json')
    parser.add_argument('--return_output', help='return output', default='./output.json')
    args = parser.parse_args() 
    main(args)