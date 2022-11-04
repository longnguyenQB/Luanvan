from processing_data.processing_function import *
import warnings
import yaml
from tqdm import tqdm
tqdm.pandas()
warnings.filterwarnings("ignore")

def get_data(data_his):
    
    with open("processing_data/external_data/external_data.yml", "r") as f:
        external_data = yaml.load(f,Loader=yaml.FullLoader)
    df = get_valid_data(data_his)
    df = features_engineering(df, external_data)
    # df.to_csv("D:/AI Project/Icaller/data/data_report2spam/data_after_clean/all_3month_by_reported_9_2022.csv")
    df = df.groupby(['report']).progress_apply(group_by_phone)
    df.reset_index(inplace=True)
    df = feature_creation(df)
    return df
