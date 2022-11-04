from datetime import datetime
from itertools import chain, product
from numpy import mean
from typing import Dict, List, Tuple
from sklearn.preprocessing import OneHotEncoder
from itertools import chain, product
from typing import Dict, List
import numpy as np
import pandas as pd


def __convert_time(time):
    return datetime.fromisoformat(str(time))

def __convert_time2day(time):
    return datetime.fromisoformat(str(time)).strftime('%Y-%m-%d')

def remove_prefix(input_string):
    """Xóa ký tự + và 84 ở đầu

    Args:
        input_string (_type_): string
    """
    def remove_prefix1(input_string):
        prefix = '+'
        if prefix and input_string.startswith(prefix):
            return input_string[len(prefix):]
        return input_string

    def remove_prefix2(input_string):
        prefix = '84'
        if prefix and input_string.startswith(prefix):
            return input_string[len(prefix):]
        return input_string

    input_string = remove_prefix1(input_string)
    input_string = remove_prefix2(input_string)
    return input_string


def get_provider(phone: str, providers_lib: Dict) -> str:
    """Lấy nhà cung cấp dịch vụ của số điện thoại cho trước

    Args:
        phone (str): số điện thoại
        providers_lib (Dict): từ điển các đầu số của mỗi nhà mạng

    Returns:
        str: tên nhà mạng
    """
    def __reverse_dict(dictionary: Dict):
        mapper = lambda k, v: product(v, [k])
        packed_object = map(mapper, dictionary.keys(), dictionary.values())
        return dict(chain.from_iterable(packed_object))

    def __provider_search(prefix: str):
        return phone_by_provider.get(prefix, 'others')

    prefix = phone[:2]
    phone_by_provider = __reverse_dict(providers_lib)
    provider = __provider_search(prefix)
    return provider


def is_same_provider(caller: str, receiver: str, providers_lib: Dict) -> bool:
    """Kiểm tra liệu người gọi và người nhận có chung mạng không

    Args:
        caller (str): số điện thoại người gọi
        receiver (str): số điện thoại người nhận
        providers_lib (Dict): từ điển các đầu số của mỗi nhà mạng

    Returns:
        bool: true/false
    """
    caller_provider = get_provider(phone=caller, providers_lib=providers_lib)
    receiver_provider = get_provider(phone=receiver,
                                     providers_lib=providers_lib)
    if caller_provider == receiver_provider:
        return True
    else:
        return False
    
def is_landline(caller:str, landline_lib ):
    """Kiểm tra người gọi là loại số cố định hay di động

    Args:
        caller (str): số điện thoại người gọi
        landline_lib (_type_): từ điển các đầu số của các SĐT cố định

    Returns:
        bool: true/false
    """    
    if caller.startswith(tuple(landline_lib)):
        return True
    else:
        return False

def is_same_country(caller, receiver, country_lib ):  
    """Kiểm tra cùng Việt Nam không

    Args:
        caller (_type_): người gọi
        receiver (_type_): người nhận
        country_lib (_type_): từ điển các đầu số của VN
    """          
    def _get_country(phone, country_lib):
        if phone.startswith(country_lib):
            return '1'
        else:
            return '0'
    caller = _get_country(caller, country_lib)
    receiver = _get_country(receiver, country_lib)
    if caller == receiver:
        return True
    else:
        return False

    
def get_valid_data(df: pd.DataFrame) -> pd.DataFrame:
    """Lọc dữ liệu hợp lệ

    Args:
        df (pd.DataFrame): Dữ liệu gốc

    Returns:
        pd.DataFrame: Dữ liệu đã lọc, chỉ giữ lại các bản ghi hợp lệ
    """
    def __is_valid_phone(phone: str) -> bool:
        """Kiểm tra tính hợp lệ của số điện thoại

        Args:
            phone (str): Số điện thoại đầu vào

        Returns:
            bool: True/False
        """
        length = len(phone)
        if length > 4 and length < 18:
            return True
        return False

    name_col = [
        'phone', 'type', 'time', 'in_contact', 'duration', 'phone_member',  'Spam',
        'report',
        # 'member_report',
        # 'time_report',
    ]
    df = df[name_col]
    df = df.dropna()
    data = df.loc[(df.type.isin([1, 2, 3, 4]))]
    data.phone = data.phone.apply(remove_prefix)
    data.phone_member = data.phone_member.apply(remove_prefix)
    data.reset_index(inplace=True)
    data = data.drop(data.loc[(data.phone.str.len() < 4) |
                              (data.phone.str.len() > 18)].index)
    alphabet = '0123456789'
    data = data[~data.phone.str.strip(alphabet).astype(bool)]
    data = data.dropna()
    data.reset_index(inplace=True)
    del data['index']

    return data


def get_dummies_variables(
    df: pd.DataFrame,
    type_map: Dict = {
        1: 'call_to',
        2: 'call_in',
        3: 'call_to_miss',
        4: 'call_in_miss'
    }
) -> pd.DataFrame:
    """Onehot encoder

    Args:
        df (pd.DataFrame): [description]
        type_map (Dict, optional): [description]. Defaults to { 1: 'call_to', 2: 'call_in', 3: 'call_to_miss', 4: 'call_in_miss' }.

    Returns:
        [type]: [description]
    """
    
    types_list = list(type_map.keys())
    columns = ['call_to', 'call_in', 'call_to_miss', 'call_in_miss']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, drop=None)
    encoder.fit([[i] for i in types_list])
    types = df.type.to_numpy().reshape(-1, 1)
    encoded_data = encoder.transform(types)
    to_concatenated = pd.DataFrame(encoded_data, columns=columns)
    df.reset_index(inplace=True)
    to_concatenated.reset_index(inplace=True)
    df = pd.concat([df, to_concatenated], axis=1)
    # df = df.drop(labels=['type'], axis=1)
    df = df.sort_values(by=['time'])
    df['duration_call_in'] = df['duration'] * df['call_in']
    df['duration_call_to'] = df['duration'] * df['call_to']
    return df


def features_engineering(df: pd.DataFrame, external_data: Dict):
    def __in_hour(hour: float,
                  day: int,
                  start_hour: int = 7,
                  end_hour: int = 19,
                  work_days: List = [2, 3, 4, 5, 6]) -> bool:
        """Kiểm tra có đang trong giờ hành chính không

        Args:
            hour (float): Thời điểm muốn kiểm tra
            low (int, optional): Giờ bắt đầu làm việc. Defaults to 7.
            high (int, optional): Giờ kết thúc làm việc. Defaults to 19.

        Returns:
            bool: [description]
        """
        is_work_hour = (hour - start_hour) * (hour - end_hour) <= 0
        is_work_day = day in work_days
        if is_work_day and is_work_hour:
            return True
        else:
            return False

    def __is_successed(duration: float):
        return True if duration > 20 else False

    providers_lib = external_data['providers']
    work_days = external_data['work_days']
    type_maps = external_data['type_map']
    landline = external_data['landline']
    country_lib = tuple(chain(*external_data['providers'].values())) + tuple(external_data['landline'].values())

    dt = pd.to_datetime(df.time).dt
    call_hours = dt.hour
    call_days = dt.dayofweek

    is_shared_providers = list(
        map(
            lambda caller, receiver: is_same_provider(
                caller=caller, receiver=receiver, providers_lib=providers_lib),
            df.phone, df.phone_member))
    is_landlines = list(
        map(
            lambda caller: is_landline(
                caller=caller, landline_lib=landline),
            df.phone))
    same_countrys = list(
        map(
            lambda caller, receiver: is_same_country(
                caller=caller, receiver=receiver, country_lib=country_lib),
            df.phone, df.phone_member))
    
    df['is_landline'] = is_landlines
    df['same_network'] = is_shared_providers
    df['same_country'] = same_countrys
    df['in_hour'] = list(
        map(
            lambda hour, day: __in_hour(hour=hour,
                                        day=day,
                                        start_hour=7,
                                        end_hour=19,
                                        work_days=work_days), call_hours,
            call_days))
    df['not_in_hour'] = ~df['in_hour']
    df['time'] = df['time'].apply(__convert_time)
    # df['time_report'] = df['time_report'].apply(__convert_time)
    df['success'] = df['duration'].apply(__is_successed)

    df = get_dummies_variables(df=df, type_map=type_maps)
    return df


"""Phần này xử lí danh sách các báo cáo từ members
"""
def total_call_before_report(df: pd.DataFrame):
    tmp = df.loc[df.phone_member == df.phone_report.values[0]]
    return tmp.call_to.sum() + tmp.call_in.sum() + tmp.call_to_miss.sum() + tmp.call_in_miss.sum()

def group_by_phone(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    def __get_frequency(time):
        freq = np.median(np.diff(time))
        freq_by_hour = (freq / np.timedelta64(1, 's')) / 3600
        return freq_by_hour

    def __get_frequency_degree_by_phone_member(df, type):
        if type == "in":
            df_in = df.loc[(df.type == 2) | (df.type == 4)]
            df_in = df_in.groupby(by=['phone_member']).agg(
                freq_to=('time', lambda x: __get_frequency(x)))
            df_in.reset_index(inplace=True)
            if df_in.empty:
                return 0
            else:
                return df_in.freq_to.sum()
        elif type == 'to':
            df_out = df.loc[(df.type == 1) | (df.type == 3)]
            df_out = df_out.groupby(by=['phone_member']).agg(
                freq_to=('time', lambda x: __get_frequency(x)))
            df_out.reset_index(inplace=True)
            if df_out.empty:
                return 0
            else:
                return df_out.freq_to.sum()

    def __get_frequency_degree(df, type):
        if type == "in":
            df_in = df.loc[(df.type == 2) | (df.type == 4)]
            freq = np.mean(np.diff(df_in.time))
            freq_by_hour = (freq / np.timedelta64(1, 's')) / 3600
            return freq_by_hour
        elif type =="to":
            df_out = df.loc[(df.type == 1) | (df.type == 3)]
            freq = np.mean(np.diff(df_out.time))
            freq_by_hour = (freq / np.timedelta64(1, 's')) / 3600
            return freq_by_hour

    def __get_timejoin(time):
        time = time.max() - time.min()
        return (time / np.timedelta64(1, 's')) / 3600
    def __get_timejoin2report(df):
        
        time = df.time_report.values[0] - df.time.min()
        return (time / np.timedelta64(1, 's')) / 3600

    def __get_total_contacted(phone_member):
        total_contacted = len(phone_member.unique())
        return total_contacted

    def _get_total_contacted_degree(df, type):
        if type == "in":
            return len(df.loc[(df.type == 2) |
                              (df.type == 4)].phone_member.unique())
        else:
            return len(df.loc[(df.type == 1) |
                              (df.type == 3)].phone_member.unique())

    def _get_redial(phone_member):
        redial = 0
        phone_member = list(phone_member)
        for i in range(len(phone_member)):
            try:
                if phone_member[i] == phone_member[i + 1]:
                    redial += 1
            except:
                return redial    
    def _get_call_by_day(df):
        stream = df.copy()
        stream.time = stream.time.apply(__convert_time2day)
        stream.groupby(by=['time']).apply(lambda df,a,b: mean(df[a] + df[b]), 'call_to', 'call_in')    
        
    def _get_info_report(df, type):
        tmp = df.loc[df.phone_member == df.member_report.values[0]]
        if type == 'total_call':
            return len(tmp)
        elif type == 'duration_with_reporter':
            return tmp.duration.mean()
        elif type == 'duration_call_to_with_reporter':
            return tmp.duration_call_to.mean()
        elif type == 'duration_call_in_with_reporter':
            return tmp.duration_call_in.mean()
        elif type == 'call_to_with_reporter':
            return tmp.call_to.sum()
        elif type == 'call_in_with_reporter':
            return tmp.call_in.sum()
        elif type == 'call_to_miss_with_reporter':
            return tmp.call_to_miss.sum()
        elif type == 'call_in_miss_with_reporter':
            return tmp.call_in_miss.sum()
        elif type == 'time_lastcall2report':
            time_max = df.time.max()
            try:
                time_created = (df.time_report - time_max)
                index_ = time_created.where(time_created >= timedelta(days=0)).sort_values().index[0]
                return time_created[index_]
            except:
                return 0 
        elif type == "stream_with_memberreport":
            stream = df.loc[df.phone_member == df.member_report.values[0]]
            if stream.call_in ==0:
                return 0
            else:
                return stream.call_to + stream.call_in
        elif type == "call_by_day":
            stream = df.loc[df.phone_member == df.member_report.values[0]]
            stream.time = stream.time.apply(__convert_time2day)
            return mean(stream.groupby(by=['time']).apply(lambda df,a,b: sum(df[a] + df[b]), 'call_to', 'call_in'))
            
                
    mapper = {}
    mapper["avg_in_contact"] = df['in_contact'].mean()
    mapper["same_network"] = df['same_network'].mean()
    mapper["duration"] = df['duration'].mean()
    mapper["duration_call_to"] = df['duration_call_to'].mean()
    mapper["duration_call_in"] = df['duration_call_in'].mean()
    mapper["in_hour"] = df['in_hour'].sum()
    mapper["not_in_hour"] = df['not_in_hour'].sum()
    mapper["avg_success"] = df['success'].mean()
    mapper["call_to"] = df['call_to'].sum()
    mapper["call_in"] = df['call_in'].sum()
    mapper["call_to_miss"] = df['call_to_miss'].sum()
    mapper["call_in_miss"] = df['call_in_miss'].sum()
    mapper["frequency"] = __get_frequency(df['time'])
    mapper["frequency_out"] = __get_frequency_degree(df, 'to')
    mapper["frequency_in"] = __get_frequency_degree(df, 'in')
    mapper[
        "frequency_out_by_phonemember"] = __get_frequency_degree_by_phone_member(
            df, 'to')
    mapper[
        "frequency_in_by_phonemember"] = __get_frequency_degree_by_phone_member(
            df, 'in')
    mapper["time_join"] = __get_timejoin(df['time'])
    mapper["total_redial"] = _get_redial(df['phone_member'])
    mapper["total_contacted"] = __get_total_contacted(df['phone_member'])
    mapper["total_contacted_to"] = _get_total_contacted_degree(df, 'to')
    mapper["total_contacted_in"] = _get_total_contacted_degree(df, 'in')
    mapper["mean_call_by_day"] = _get_call_by_day(df)
    # mapper["time_join2report"] = __get_timejoin2report(df)
    # mapper["time_lastcall2report"] = _get_info_report(df, 'time_lastcall2report')
    # mapper["total_call_with_reporter"] = _get_info_report(df, 'total_call')
    # mapper["duration_with_reporter"] = _get_info_report(df, 'duration_with_reporter')
    # mapper["duration_call_to_with_reporter"] = _get_info_report(df, 'duration_call_to_with_reporter')
    # mapper["duration_call_in_with_reporter"] = _get_info_report(df, 'duration_call_in_with_reporter')
    # mapper["call_to_with_reporter"] = _get_info_report(df, 'call_to_with_reporter')
    # mapper["call_in_with_reporter"] = _get_info_report(df, 'call_in_with_reporter')
    # mapper["call_to_miss_with_reporter"] = _get_info_report(df, 'call_to_miss_with_reporter')
    # mapper["call_in_miss_with_reporter"] = _get_info_report(df, 'call_in_miss_with_reporter')
    # mapper["stream_with_memberreport"] = _get_info_report(df, 'stream_with_memberreport')
    # mapper["call_by_day"] = _get_info_report(df, 'call_by_day')
    mapper["Spam"] = df['Spam'].max()
    return pd.Series(
        mapper,
        index=[
            'avg_in_contact', 'same_network', 'duration', 'duration_call_to',
            "duration_call_in", "in_hour", "not_in_hour", "avg_success",
            "call_to", "call_in", "call_to_miss", "call_in_miss", "frequency",
            "frequency_out", "frequency_in",
            "frequency_out_by_phonemember",
            "frequency_in_by_phonemember", "time_join", "time_join2report" , "total_redial",
            "total_contacted", "total_contacted_to", "total_contacted_in", "mean_call_by_day",
            # "time_lastcall2report", "total_call_with_reporter", "duration_with_reporter", "duration_call_to_with_reporter", "duration_call_in_with_reporter", "call_to_with_reporter", "call_in_with_reporter", "call_to_miss_with_reporter","call_in_miss_with_reporter", 'stream_with_memberreport', 'call_by_day',
            "Spam"
        ])


# Sinh các feature mới cho train test
def feature_creation(df: pd.DataFrame) -> pd.DataFrame:
    """ Sinh các feature mới

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    def zero_div(x, y):
        try:
            return x / y
        except ZeroDivisionError:
            return 0

    df['call_to_rate'] = zero_div(
        df.call_to,
        (df.call_to + df.call_in + df.call_to_miss + df.call_in_miss))
    df['call_in_rate'] = zero_div(
        df.call_in,
        (df.call_to + df.call_in + df.call_to_miss + df.call_in_miss))
    df['call_to_miss_rate'] = zero_div(
        df.call_to_miss,
        (df.call_to + df.call_in + df.call_to_miss + df.call_in_miss))
    df['call_in_miss_rate'] = zero_div(
        df.call_in_miss,
        (df.call_to + df.call_in + df.call_to_miss + df.call_in_miss))
    df['callto_callin'] = zero_div((df.call_to), (df.call_in))
    df['call_back_rate'] = zero_div((df.call_to_miss), (df.call_in))
    df['call_miss'] = zero_div((df.call_to + df.call_in),
                               (df.call_in_miss + df.call_to_miss))
    df['sum_call'] = df.call_in + df.call_to + df.call_in_miss + df.call_to_miss
    df['avg_duration_call_to'] = zero_div(df.duration_call_to, (df.call_to))
    df['avg_duration_call_in'] = zero_div(df.duration_call_in, (df.call_in))
    df['duration_div_call_to_in'] = zero_div(df.duration,
                                             (df.call_to + df.call_in))
    # df = df.loc[df.duration_div_call_to_in != 0]
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df = df.fillna(0)

    df = df.astype({
        'call_to': 'int32',
        'call_in': 'int32',
        'call_to_miss': 'int32',
        'call_in_miss': 'int32',
        'duration_call_to': 'int32',
        'duration_call_in': 'int32'
    })

    # df = df.drop(index= df.loc[(df.time_join == 0) | (df.time_join == df.frequency)].index)
    df['call_to_miss_mul_frequency'] = df['call_to'] * df['frequency']
    df['duration_call_to_rate'] = df['duration_call_to'] / (
        df['duration_call_to'] + df['duration_call_in'] + 1)
    df['call_to_mul_in_contact'] = df['call_to'] * df['avg_in_contact']
    df['call_to_mul_success'] = df['call_to'] * df['avg_success']
    df['frequency_mul_in_hour'] = df['frequency'] * df['in_hour']
    df['call_to_miss_frequency_in_hour'] = df['call_in_miss_rate'] * df[
        'frequency_mul_in_hour']
    df['call_in_div_call_to'] = (df['call_in'] + df['call_in_miss']) / (
        df['call_to'] + df['call_in_miss'] + 1)
    df['call_in_div_call_to_mul_in_hour'] = df['call_in_div_call_to'] * df[
        'in_hour']
    df['call_to_miss_rate_mul_duration_call_to'] = df[
        'call_to_miss_rate'] * df['duration_call_to']
    df['frequency_in_hour_mul_duration_call_to'] = df[
        'frequency_mul_in_hour'] * df['duration_call_to']
    df['in_hour_mul_avg_success'] = df['in_hour'] * df['avg_success']
    df['call_to_miss_rate_mul_duration_call_to'] = df[
        'call_to_miss_rate'] * df['duration_call_to']
    df['call_in_div_call_to_mul_in_hour_mul_avg_success'] = df[
        'frequency_in_hour_mul_duration_call_to'] * df['call_in_div_call_to']
    
    # del df['call_to']
    # del df['call_in']
    # del df['call_to_miss']
    # del df['call_in_miss']
    return df


"""Phần này xử lí danh sách các báo cáo từ members
"""


def get_report_data(df: pd.DataFrame) -> pd.DataFrame:
    pass
