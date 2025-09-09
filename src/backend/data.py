#  data upload (upload csv, exal, or json file)
import pandas as pd

def read_file (file) :
    if file.endwith('.csv'):
        df= pd.read_csv(file)
    elif file.endwith('.xlsx') or file.endswith('.xls'):
        df = pd.read_excel(file)
    elif file.endwith('.json') :
        df = pd.read_json(file)
    else:
        raise ValueError('unsupported file format')
    return df

#Data Cleaning
#handiling missing values & duplicates
