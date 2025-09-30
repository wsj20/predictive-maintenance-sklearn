import pandas as pd
import numpy as np

#NB: Data has no missing values as checked in DW
def load_data():    
    df = pd.read_csv('../data/ai4i2020.csv')
    print(f"Successfully loaded data into DF \n")
    return df

def data_overview(df):
    print(df.head())
    print(df.info())
    print(df.describe())

def get_failure_percentage(df):
    failure_percentage = df['Machine failure'].value_counts(normalize=True)
    failure_result = failure_percentage[1] * 100
    return failure_result



if __name__ == "__main__":
    df = load_data()

    failure_result = get_failure_percentage(df)
    print(f"The percentage of machine failures is: {failure_result:.2f}%")
    