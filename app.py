import pandas as pd
import numpy as np
import time
import turicreate as tc
import sklearn
import sys
from flask import Flask
import requests
from threading import Thread

from sklearn.model_selection import train_test_split


# variables to define field names
user_id = 'customerId'
item_id = 'productId'
target = 'purchase_count'
n_rec = 10 # number of items to recommend
n_display = 30

def create_data_dummy(data):
    data_dummy = data.copy()
    data_dummy['purchase_dummy'] = 1
    return data_dummy

def normalize_data(data):
    df_matrix = pd.pivot_table(data, values='purchase_count', index='customerId', columns='productId')
    df_matrix_norm = (df_matrix-df_matrix.min())/(df_matrix.max()-df_matrix.min())
    d = df_matrix_norm.reset_index()
    d.index.names = ['scaled_purchase_freq']
    return pd.melt(d, id_vars=['customerId'], value_name='scaled_purchase_freq').dropna()

def split_data(data):
    '''
    Splits dataset into training and test set.
    
    Args:
        data (pandas.DataFrame)
        
    Returns
        train_data (tc.SFrame)
        test_data (tc.SFrame)
    '''
    train, test = train_test_split(data, test_size = .2)
    train_data = tc.SFrame(train)
    test_data = tc.SFrame(test)
    return train_data, test_data

def create_output(model, users_to_recommend, n_rec, print_csv=True):
    recomendation = model.recommend(users=users_to_recommend, k=n_rec)
    df_rec = recomendation.to_dataframe()
    df_rec['recommendedProducts'] = df_rec.groupby([user_id])[item_id] \
        .transform(lambda x: '|'.join(x.astype(str)))
    df_output = df_rec[['customerId', 'recommendedProducts']].drop_duplicates() \
        .sort_values('customerId').set_index('customerId')
    # if print_csv:
    #     df_output.to_csv('option1_recommendation.csv')
    #     print("An output file can be found in 'output' folder with name 'option1_recommendation.csv'")
    return df_output

def customer_recomendation(customer_id,df_output):
    if customer_id not in df_output.index:
        print('Customer not found.')
        return customer_id
    return df_output.loc[customer_id]

def run_ml_code():
    URL = "https://alowisindiaprivatelimited.com/shop/recommendation/getData"
    r = requests.get(url = URL)
    json_data = r.json()
    customers = pd.DataFrame(json_data['customers'])
    transactions = pd.DataFrame(json_data['txn'])
    users_to_recommend = list(transactions[user_id])
    
    transactions.head()

    transactions['products'] = transactions['products'].apply(lambda x: [int(i) for i in str(x).split('|')])
    transactions.head(2).set_index('customerId')['products'].apply(pd.Series).reset_index()

    s=time.time()

    data = pd.melt(transactions.set_index('customerId')['products'].apply(pd.Series).reset_index(), 
                id_vars=['customerId'],
                value_name='products') \
        .dropna().drop(['variable'], axis=1) \
        .groupby(['customerId', 'products']) \
        .agg({'products': 'count'}) \
        .rename(columns={'products': 'purchase_count'}) \
        .reset_index() \
        .rename(columns={'products': 'productId'})
    data['productId'] = data['productId'].astype(np.int64)


    data_dummy = create_data_dummy(data)

    df_matrix = pd.pivot_table(data, values='purchase_count', index='customerId', columns='productId')
    df_matrix.head()

    df_matrix_norm = (df_matrix-df_matrix.min())/(df_matrix.max()-df_matrix.min())
    print(df_matrix_norm.shape)
    df_matrix_norm.head()

    # create a table for input to the modeling

    d = df_matrix_norm.reset_index()
    d.index.names = ['scaled_purchase_freq']
    data_norm = pd.melt(d, id_vars=['customerId'], value_name='scaled_purchase_freq').dropna()
    data_norm.head()



    train, test = train_test_split(data, test_size = .2)

    # Using turicreate library, we convert dataframe to SFrame - this will be useful in the modeling part

    train_data = tc.SFrame(train)
    test_data = tc.SFrame(test)



    train_data_dummy, test_data_dummy = split_data(data_dummy)
    train_data_norm, test_data_norm = split_data(data_norm)

    



    train.groupby(by=item_id)['purchase_count'].mean().sort_values(ascending=False).head(20)


    name = 'pearson'
    target = 'scaled_purchase_freq'

    users_to_recommend = list(customers[user_id])

    final_model = tc.item_similarity_recommender.create(tc.SFrame(data_dummy), 
                                                user_id=user_id, 
                                                item_id=item_id, 
                                                target='purchase_dummy', 
                                                similarity_type='cosine')


    df_output = create_output(final_model, users_to_recommend, n_rec, print_csv=False)
    print(df_output.shape)
    df_output.head()
    url = 'https://alowisindiaprivatelimited.com/shop/recommendation/storeResults'
    res=requests.post(url,data={"data":df_output.to_json()})
    print(res.text)
    


app = Flask(__name__)

@app.route("/")
def home_view():
        Thread(target = run_ml_code).start()
        return "1"