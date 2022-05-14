# %%
# time is our best friend の実験
# コールドスタートをオンラインの1回目の購入に限定した

# %%
import os
EXP = os.path.splitext(os.path.basename(__file__))[0]

# %%
import numpy as np
import pandas as pd

from pathlib import Path

data_path = Path('../input/h-and-m-personalized-fashion-recommendations/')

# %%
transactions = pd.read_csv(
    data_path / 'transactions_train.csv',
    # set dtype or pandas will drop the leading '0' and convert to int
    dtype={'article_id': str} 
)

submission = pd.read_csv(data_path / 'sample_submission.csv')

# %%
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
# transactions = transactions.query('t_dat <= @pd.to_datetime("2020-09-15")')
print(transactions['t_dat'].max())

# %%
transactions = transactions.drop_duplicates(subset=['t_dat', 'customer_id', 'article_id'])

# %%
transactions['t_diff'] = (transactions['t_dat'].max() - transactions['t_dat']).dt.days
transactions_3w = transactions[transactions['t_diff'] <= 20].copy()
transactions_2w = transactions[transactions['t_diff'] <= 13].copy()
transactions_1w = transactions[transactions['t_diff'] <= 6].copy()

# %%
purchase_dict_3w = {}

for i,x in enumerate(zip(transactions_3w['customer_id'], transactions_3w['article_id'])):
    cust_id, art_id = x
    if cust_id not in purchase_dict_3w:
        purchase_dict_3w[cust_id] = {}
    
    if art_id not in purchase_dict_3w[cust_id]:
        purchase_dict_3w[cust_id][art_id] = 0
    
    purchase_dict_3w[cust_id][art_id] += 1
    
print(len(purchase_dict_3w))

dummy_list_3w = list((transactions_3w['article_id'].value_counts()).index)[:12]

# %%
purchase_dict_2w = {}

for i,x in enumerate(zip(transactions_2w['customer_id'], transactions_2w['article_id'])):
    cust_id, art_id = x
    if cust_id not in purchase_dict_2w:
        purchase_dict_2w[cust_id] = {}
    
    if art_id not in purchase_dict_2w[cust_id]:
        purchase_dict_2w[cust_id][art_id] = 0
    
    purchase_dict_2w[cust_id][art_id] += 1
    
print(len(purchase_dict_2w))

dummy_list_2w = list((transactions_2w['article_id'].value_counts()).index)[:12]

# %%
purchase_dict_1w = {}

for i,x in enumerate(zip(transactions_1w['customer_id'], transactions_1w['article_id'])):
    cust_id, art_id = x
    if cust_id not in purchase_dict_1w:
        purchase_dict_1w[cust_id] = {}
    
    if art_id not in purchase_dict_1w[cust_id]:
        purchase_dict_1w[cust_id][art_id] = 0
    
    purchase_dict_1w[cust_id][art_id] += 1
    
print(len(purchase_dict_1w))

dummy_list_1w = list((transactions_1w['article_id'].value_counts()).index)[:12]

# %%
not_so_fancy_but_fast_benchmark = submission[['customer_id']]
prediction_list = []

# %%
# dummy_list = list((transactions_1w['article_id'].value_counts()).index)[:12]
online_transaction = transactions_1w[transactions_1w['sales_channel_id']==2]
# Calculate the order of the transcation per customer
order_number = online_transaction[["t_dat", "customer_id"]].groupby(["t_dat", "customer_id"]).count()
order_number.reset_index(["t_dat", "customer_id"], inplace=True)
order_number['nth_order'] = order_number.sort_values(["customer_id",'t_dat'], ascending=True)\
             .groupby(['customer_id'])\
             .cumcount() + 1
online_transaction = pd.merge(online_transaction, order_number[['t_dat', 'customer_id', 'nth_order']], how='left')
dummy_list = online_transaction[online_transaction['nth_order']==1]['article_id'].value_counts()[:12].index.values
dummy_pred = ' '.join(dummy_list)
print(dummy_pred)

# %%
for i, cust_id in enumerate(submission['customer_id'].values.reshape((-1,))):
    if cust_id in purchase_dict_1w:
        l = sorted((purchase_dict_1w[cust_id]).items(), key=lambda x: x[1], reverse=True)
        l = [y[0] for y in l]
        if len(l)>12:
            s = ' '.join(l[:12])
        else:
            s = ' '.join(l+dummy_list_1w[:(12-len(l))])
    elif cust_id in purchase_dict_2w:
        l = sorted((purchase_dict_2w[cust_id]).items(), key=lambda x: x[1], reverse=True)
        l = [y[0] for y in l]
        if len(l)>12:
            s = ' '.join(l[:12])
        else:
            s = ' '.join(l+dummy_list_2w[:(12-len(l))])
    elif cust_id in purchase_dict_3w:
        l = sorted((purchase_dict_3w[cust_id]).items(), key=lambda x: x[1], reverse=True)
        l = [y[0] for y in l]
        if len(l)>12:
            s = ' '.join(l[:12])
        else:
            s = ' '.join(l+dummy_list_3w[:(12-len(l))])
    else:
        s = dummy_pred
    prediction_list.append(s)

not_so_fancy_but_fast_benchmark['prediction'] = prediction_list
print(not_so_fancy_but_fast_benchmark.head(3))

# %%
not_so_fancy_but_fast_benchmark.to_csv(f'../submissions/{EXP}_submission.csv', index=False)
