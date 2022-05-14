#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# ランク学習
# 学習データ5週分(検証データ1週含む)
# 候補作り12週分
# trendingで候補作り
# articlesとcustomersの特徴量をtarget_weekごとに作る
# 候補の良さを測る
# 顧客毎の最終週で候補作り
# コラボ商品で候補作り
# 説明文KNN(tsne, 重複除外なし)
# ペア商品
# コラボのtarget_week=-1修正したやつ
# albertでKNN
# 説明文KNNのtsneのcos距離の掛け算を修正する
# BertでKNN
# VGGでKNN（メモリ制約のため一部ルールベース）：除外
# 購入履歴Word2VecでKNN(tsne)
# customerの年齢カラムでbinningしてpopular items
# Bert4Rec：除外
# Gru4Rec：除外
# Optunaでチューニング
# VGGでKNN（全画像利用）：除外
# tsneからBertを抜いて非圧縮で独立させる
# KNNで自分自身を抜く
# 良い手法を広げる（bin popular, tsne, 圧縮word2vec）
# 他の良い手法も広げる（pair, albert）
# Bert非圧縮除外、tsneはtf-idfとbert
# KNN自分自身を抜かない
# New: 学習率落とす
# MAP@12 (all): 0.033614
# MAP@12 (cold start): 0.008811

# from 055
EXP = '058'


# %%


from pathlib import Path
import pickle
import gc
import os
from time import time
import warnings
from functools import reduce
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from psutil import cpu_count

from line_notify import send_line_notification

tqdm.pandas()
pd.options.display.max_columns = None
warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
warnings.simplefilter('ignore', UserWarning)
data_path = Path('../input/h-and-m-personalized-fashion-recommendations/')


# %%


send_line_notification(f'[EXP{EXP}] start.')
transactions = pd.read_csv(
    data_path / f'transactions_train.csv',
    # set dtype or pandas will drop the leading '0' and convert to int
    dtype={'article_id': 'int32'},
    parse_dates=['t_dat'])
customers = pd.read_csv(data_path / 'customers.csv')
articles = pd.read_csv(
    data_path / 'articles.csv', dtype={'article_id': 'int32'})

t_max = transactions['t_dat'].max()
transactions['t_diff'] = (t_max - transactions['t_dat']).dt.days
transactions['week'] = transactions['t_diff'] // 7

# Noneの表記不揃い対策
customers.loc[~customers['fashion_news_frequency'].isin(['Regularly', 'Monthly']), 'fashion_news_frequency'] = None

# メモリ削減
id_to_index_dict = dict(zip(customers["customer_id"], customers.index))
index_to_id_dict = dict(zip(customers.index, customers["customer_id"]))
transactions["customer_id"] = transactions["customer_id"].map(id_to_index_dict).astype('int32')
customers['customer_id'] = customers['customer_id'].map(id_to_index_dict).astype('int32')


# %%


def generate_candidates_trending(transactions: pd.DataFrame, customers: np.ndarray, target_week: int):
    # week列のシフト(week=target_week => week=0)、リーク防止
    df = transactions.query("customer_id in @customers").copy()
    df['week'] = df['week'] - target_week
    df = df.query('week >= 1')
    
    # 以下のロジックはtrendingの公開カーネル参照
    weekly_sales = df.groupby(['week', 'article_id'])['article_id'].count().rename('count').reset_index()
    df = df.merge(weekly_sales, on=['week', 'article_id'], how='left')    
    weekly_sales = weekly_sales.reset_index().set_index('article_id')
    df = df.join(weekly_sales.loc[weekly_sales['week']==1, ['count']], on='article_id', rsuffix='_targ')
    df['count_targ'].fillna(0, inplace=True)
    df['quotient'] = df['count_targ'] / df['count']
    
    t_max = df['t_dat'].max()
    df['x'] = ((t_max - df['t_dat']) / np.timedelta64(1, 'D')).astype(int)
    df['dummy_1'] = 1
    df['x'] = df[['x', 'dummy_1']].max(axis=1)    
    a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
    df['y'] = a / np.sqrt(df['x']) + b * np.exp(-c*df['x']) - d
    df['dummy_0'] = 0 
    df['y'] = df[["y", "dummy_0"]].max(axis=1)
    df['trending_value'] = df['quotient'] * df['y']
    df = df.groupby(['customer_id', 'article_id']).agg({'trending_value': 'sum'}).reset_index()
    df = df.loc[df['trending_value'] > 0]
    # df['rank'] = df.groupby("customer_id")["value"].rank("dense", ascending=False)
    # df = df.loc[df['rank'] <= 12]
    
    df['isin_trending'] = 1

    return df[['customer_id', 'article_id', 'trending_value', 'isin_trending']]


# %%


def generate_candidates_recently(transactions: pd.DataFrame, customers: np.ndarray, target_week: int):
    '''直近の購入履歴から候補生成'''
    # week列のシフト(week=target_week => week=0)、リーク防止
    df = transactions.query("customer_id in @customers").copy()
    df['week'] = df['week'] - target_week
    df = df.query("week >= 1")
    df = df.query("week <= 12")  # 12週分の履歴を使用
    
    # 各週の購入個数から特徴量
    for w in df['week'].unique()[::-1]:
        tmp = df.query('week == @w').groupby(['customer_id', 'article_id'])['article_id'].count().rename(f'count_{w}w').fillna(0).reset_index().copy()
        if w == 1:
            purchase_df = tmp
            continue
        purchase_df = purchase_df.merge(tmp, how='outer', on=['customer_id', 'article_id'])
        purchase_df[f'count_{w}w'] = purchase_df[f'count_{w}w'].fillna(0)
        
    purchase_df['isin_recently'] = 1
    
    return purchase_df


# %%


def generate_candidates_popular(transactions: pd.DataFrame, customers: np.ndarray, target_week: int):
    '''販売数の多い商品から候補生成'''
    # make_articles_featureが販売数から特徴量を作ってくれるから、この関数は候補のペアだけ返す
    
    # week列のシフト(week=target_week => week=0)、リーク防止
    df = transactions.copy()
    df['week'] = df['week'] - target_week
    df = df.query("week >= 1")
    df = df.query("week <= 4")  # 4週分の履歴を使用

    # 各週の販売数上位12個
    dummy_count_df = df.groupby(['article_id', 'week'])['week'].count().rename('dummy_count').reset_index().copy()
    dummy_count_df['rank_in_week'] = dummy_count_df.groupby('week')['dummy_count'].rank(method='min', ascending=False)
    dummy_articles = dummy_count_df.query('rank_in_week <= 12')['article_id'].unique()

    dummy_df = pd.DataFrame(
        np.concatenate(
            [np.repeat(customers, repeats=len(dummy_articles)).reshape(-1, 1),
            np.repeat(dummy_articles[None, :], repeats=len(customers), axis=0).reshape(-1, 1)],
            axis=1),
        columns = ['customer_id', 'article_id']
    )
    
    dummy_df['isin_popular'] = 1
    
    return dummy_df

def generate_candidates_bin_popular(transactions: pd.DataFrame, customers: np.ndarray, target_week: int):
    '''年齢でbinningして販売数の多い商品から候補生成'''
    # week列のシフト(week=target_week => week=0)、リーク防止
    df = transactions.copy()
    df['week'] = df['week'] - target_week
    df = df.query("week >= 1")
    df = df.query("week <= 4")  # 4週分の履歴を使用

    customers_df = pd.read_csv(data_path / 'customers.csv')
    customers_df['customer_id'] = customers_df['customer_id'].map(id_to_index_dict).astype('int32')
    customers_df['age_bin'] = pd.cut(customers_df['age'], bins=[10, 20, 30, 40, 50, 60, 70, 100], labels=False)
    
    df = df.merge(customers_df[['customer_id', 'age', 'age_bin']], how='left')
    
    bin_popular_items = df.groupby('age_bin')['article_id'].value_counts().rename('count_groupby_bin').reset_index()
    bin_popular_items['rank'] = bin_popular_items.groupby('age_bin')['count_groupby_bin'].rank('min', ascending=False)
    bin_popular_items = bin_popular_items[bin_popular_items['rank'] <= 100]
    bin_popular_items = bin_popular_items.drop('rank', axis=1)

    popular_df = pd.merge(customers_df.query("customer_id in @customers"), bin_popular_items, how='inner', on='age_bin')
    
    popular_df['isin_bin_popular'] = 1
    
    return popular_df[['customer_id', 'article_id', 'age_bin', 'count_groupby_bin', 'isin_bin_popular']]


# %%


def generate_candidates_lastw(transactions: pd.DataFrame, customers: np.ndarray, target_week: int):
    '''customer毎の最後の購入から1週間の購入履歴から候補生成'''
    # week列のシフト(week=target_week => week=0)、リーク防止
    df = transactions.query("customer_id in @customers").copy()
    df['week'] = df['week'] - target_week
    df = df.query("week >= 1").copy()

    # 最後の購入から1週間に絞る
    df['max_dat'] = df['customer_id'].map(df.groupby('customer_id')['t_dat'].max())
    df['max_diff'] = (df['max_dat'] - df['t_dat']).dt.days
    df = df.query('max_diff <= 6')
    
    df = df.merge(df.groupby(['customer_id', 'article_id'])['article_id'].count().rename('count_lastw').reset_index(), on=['customer_id', 'article_id'])
    df = df.sort_values('t_dat', ascending=True)
    df = df.drop_duplicates(['customer_id', 'article_id'], keep='last')
    df = df.rename(columns={'t_dat': 'last_dat'})
    df['last_dat'] = df['last_dat'].view(np.int64) // 10 ** 9
    
    df['isin_lastw'] = 1
    
    return df[['customer_id', 'article_id', 'count_lastw', 'last_dat', 'isin_lastw']]


# %%


def generate_candidates_pair(transactions: pd.DataFrame, customers: np.ndarray, target_week: int):
    '''ルールベース協調フィルタリングから候補生成'''
    # week列のシフト(week=target_week => week=0)、リーク防止
    df = transactions.query("customer_id in @customers").copy()
    df['week'] = df['week'] - target_week
    df = df.query("week >= 1")
    df = df.query('week <= 4').copy()  # 4週分の履歴を使用
    
    df = df.merge(df.groupby(['customer_id', 'article_id'])['article_id'].count().rename('count').reset_index(), on=['customer_id', 'article_id'])
    df = df.drop_duplicates(['customer_id', 'article_id'], keep='last')
    
    n_neighbors = 20
    pairs = pd.read_parquet(f'../input/hmitempairs/pairs_fold{target_week}_{n_neighbors}items.parquet')
    df = df.merge(pairs, on='article_id', how='inner')
    df = df.drop('article_id', axis=1).rename(columns={'pair_article_id': 'article_id'})
    df['count_pair'] = df['count'] * df['pair_ratio']

    df['isin_pair'] = 1

    return df[['customer_id', 'article_id', 'count_pair', 'isin_pair']]


# %%


def generate_candidates_collabo(transactions: pd.DataFrame, customers: np.ndarray, target_week: int):
    '''コラボ商品の購入履歴から候補生成'''
    df = pd.read_csv('../input/ranking_features/collabo_candidate_with_nseries.csv', dtype={'article_id': 'int32'})
    # FIXME: ほんとはスコープ外変数の参照は良くない
    df['customer_id'] = df['customer_id'].map(id_to_index_dict).astype('int32')
    df = df.query('customer_id in @customers').copy()
    df = df.query('target_week == @target_week').copy()
    
    df['isin_collabo'] = 1

    return df


# %%


def _generate_candidates_desc_knn(transactions: pd.DataFrame, customers: np.ndarray, target_week: int, encoder: str):
    '''説明文の埋め込みベクトル（次元削減済み）から候補生成'''
    # week列のシフト(week=target_week => week=0)、リーク防止
    df = transactions.query("customer_id in @customers").copy()
    df['week'] = df['week'] - target_week
    df = df.query("week >= 1")
    df = df.query('week <= 4').copy()  # 4週分の履歴を使用
    df = df.groupby(['customer_id', 'article_id'])['article_id'].count().rename('count').reset_index()
    
    n_neighbors = 20
    desc_knn = pd.read_csv(f'../input/ranking_features/item_features_{encoder}_class_tfidf_bert_{n_neighbors}items.csv', dtype={'article_id': 'int32'})
    
    dfs = []
    for i in range(n_neighbors):
        dfs.append(df.merge(desc_knn[['article_id', f'knn_article_id_{i}', f'knn_distance_{i}']])
                        .rename(columns={f'knn_article_id_{i}': 'knn_article_id', f'knn_distance_{i}': f'{encoder}_distance'}))
    df = pd.concat(dfs, axis=0, ignore_index=True)
    # df = df[df['article_id'] != df['knn_article_id']]  # 自分自身を削除
    df = df.drop('article_id', axis=1).rename(columns={'knn_article_id': 'article_id'})
    df[f'{encoder}_count'] = df['count'] * abs(1-df[f'{encoder}_distance'])
    
    # # 重複を削除、item1とitem2それぞれの類似として同じitemが出てくることがある
    # df = df.sort_values(f'{encoder}_count', ascending=True).drop_duplicates(subset=['customer_id', 'article_id'], keep='first')
    
    df[f'isin_{encoder}_knn'] = 1
                   
    return df[['customer_id', 'article_id', f'{encoder}_count', f'{encoder}_distance', f'isin_{encoder}_knn']]

def _generate_candidates_albert_knn(transactions: pd.DataFrame, customers: np.ndarray, target_week: int):
    # week列のシフト(week=target_week => week=0)、リーク防止
    df = transactions.query("customer_id in @customers").copy()
    df['week'] = df['week'] - target_week
    df = df.query("week >= 1")
    df = df.query('week <= 4').copy()  # 4週分の履歴を使用
    df = df.groupby(['customer_id', 'article_id'])['article_id'].count().rename('count').reset_index()
    
    n_neighbors = 20
    albert_knn = pd.read_csv(f'../input/ranking_features/Albert_cos_{n_neighbors}items.csv', dtype={'article': 'int32', 'knn_article': 'int32'})
    albert_knn = albert_knn.rename(columns={'article': 'article_id'})
    
    df = df.merge(albert_knn)
    # df = df[df['article_id'] != df['knn_article']]  # 自分自身を削除
    df = df.drop('article_id', axis=1).rename(columns={'knn_article': 'article_id', 'distances': 'albert_distance'})
    df['albert_count'] = df['count'] * abs(1-df['albert_distance'])
    
    df['isin_albert_knn'] = 1
    
    return df[['customer_id', 'article_id', 'albert_count', 'albert_distance', 'isin_albert_knn']]

def _generate_candidates_bert_knn(transactions: pd.DataFrame, customers: np.ndarray, target_week: int):
    # week列のシフト(week=target_week => week=0)、リーク防止
    df = transactions.query("customer_id in @customers").copy()
    df['week'] = df['week'] - target_week
    df = df.query("week >= 1")
    df = df.query('week <= 4').copy()  # 4週分の履歴を使用
    df = df.groupby(['customer_id', 'article_id'])['article_id'].count().rename('count').reset_index()

    usecols=['article_id', 'knn_article_id_0', 'knn_distance_0', 'knn_article_id_1', 'knn_distance_1', 
             'knn_article_id_2', 'knn_distance_2', 'knn_article_id_3', 'knn_distance_3', 'knn_article_id_4', 'knn_distance_4']
    bert_knn = pd.read_csv('../input/ranking_features/article_bert_pairs.csv', dtype={'article_id': 'int32'}, usecols=usecols)

    dfs = []
    for i in range(5):
        dfs.append(df.merge(bert_knn[['article_id', f'knn_article_id_{i}', f'knn_distance_{i}']])
                        .rename(columns={f'knn_article_id_{i}': 'knn_article_id', f'knn_distance_{i}': 'bert_distance'}))
    df = pd.concat(dfs, axis=0, ignore_index=True)
    # df = df[df['article_id'] != df['knn_article_id']]  # 自分自身を削除
    df = df.drop('article_id', axis=1).rename(columns={'knn_article_id': 'article_id'})
    df['bert_count'] = df['count'] * abs(1-df['bert_distance'])
    
    df['isin_bert_knn'] = 1
    
    return df[['customer_id', 'article_id', 'bert_count', 'bert_distance', 'isin_bert_knn']]    
    
def generate_candidates_desc_knn(transactions: pd.DataFrame, customers: np.ndarray, target_week: int):
    '''商品説明文から候補生成'''
    desc_knns = []
    
    desc_knns.append(_generate_candidates_desc_knn(transactions, customers, target_week, encoder='tsne'))
    # desc_knns.append(_generate_candidates_desc_knn(transactions, customers, target_week, encoder='umap'))
    desc_knns.append(_generate_candidates_albert_knn(transactions, customers, target_week))
    # desc_knns.append(_generate_candidates_bert_knn(transactions, customers, target_week))
    
    df = reduce(lambda left, right: pd.merge(left, right, how='outer', on=['customer_id', 'article_id']), desc_knns)

    return df


# %%


def generate_candidates_im_knn(transactions: pd.DataFrame, customers: np.ndarray, target_week: int):
    '''画像の埋め込みベクトルから候補生成'''
    # week列のシフト(week=target_week => week=0)、リーク防止
    df = transactions.query("customer_id in @customers").copy()
    df['week'] = df['week'] - target_week
    df = df.query("week >= 1")
    df = df.query('week <= 4').copy()  # 4週分の履歴を使用
    df = df.groupby(['customer_id', 'article_id'])['article_id'].count().rename('count').reset_index()
    
    im_knn = pd.read_parquet(f'../input/ranking_features/candidates_knn_vgg.parquet')
    im_knn['article_id'] = im_knn['article_id'].astype('int32')
    im_knn['knn_article_id'] = im_knn['knn_im_article_id'].astype('int32')
    
    df = df.merge(im_knn)
    # df = df[df['article_id'] != df['knn_article_id']]  # 自分自身を削除
    df = df.drop('article_id', axis=1).rename(columns={'knn_article_id': 'article_id', 'knn_distance': 'vgg_distance'})
    df[f'vgg_count'] = df['count'] * abs(1-df['vgg_distance'])
    
    # # 重複を削除、item1とitem2それぞれの類似として同じitemが出てくることがある
    # df = df.sort_values(f'{encoder}_count', ascending=True).drop_duplicates(subset=['customer_id', 'article_id'], keep='first')
    
    df[f'isin_vgg_knn'] = 1
                   
    return df[['customer_id', 'article_id', 'vgg_count', 'vgg_distance', 'isin_vgg_knn']]


def generate_candidates_sequence_word2vec(transactions: pd.DataFrame, customers: np.ndarray, target_week: int):
    '''購入履歴をシーケンスと見立て、article_idをword2vec'''
    # week列のシフト(week=target_week => week=0)、リーク防止
    df = transactions.query("customer_id in @customers").copy()
    df['week'] = df['week'] - target_week
    df = df.query("week >= 1")
    df = df.query('week <= 4').copy()  # 4週分の履歴を使用
    df = df.groupby(['customer_id', 'article_id'])['article_id'].count().rename('count').reset_index()
    
    n_neighbors = 20
    desc_knn = pd.read_csv(f'../input/ranking_features/item_features_tsne_word2vec_date_{n_neighbors}items.csv', dtype={'article_id': 'int32'})
    
    dfs = []
    for i in range(n_neighbors):
        dfs.append(df.merge(desc_knn[['article_id', f'knn_article_id_{i}', f'knn_distance_{i}']])
                        .rename(columns={f'knn_article_id_{i}': 'knn_article_id', f'knn_distance_{i}': 'sequence_word2vec_distance'}))
    df = pd.concat(dfs, axis=0, ignore_index=True)
    # df = df[df['article_id'] != df['knn_article_id']]  # 自分自身を削除
    df = df.drop('article_id', axis=1).rename(columns={'knn_article_id': 'article_id'})
    df['sequence_word2vec_count'] = df['count'] * df['sequence_word2vec_distance']
    
    # # 重複を削除、item1とitem2それぞれの類似として同じitemが出てくることがある
    # df = df.sort_values(f'{encoder}_count', ascending=True).drop_duplicates(subset=['customer_id', 'article_id'], keep='first')
    
    df['isin_sequence_word2vec_knn'] = 1
                   
    return df[['customer_id', 'article_id', 'sequence_word2vec_count', 'sequence_word2vec_distance', 'isin_sequence_word2vec_knn']]


def _generate_candidates_lstm(transactions: pd.DataFrame, customers: np.ndarray, target_week: int, model: str):
    df = pd.read_parquet(f'../input/ranking_features/{model}_fold{target_week}.parquet')
    df['customer_id'] = df['customer_id'].map(id_to_index_dict).astype('int32')
    df = df[df['customer_id'].isin(customers)]
    df = df.rename(columns={'prediction': 'article_id'})
    df['article_id'] = df['article_id'].astype('int32')
    
    df['isin_lstm'] = 1
    
    return df

def generate_candidates_lstm(transactions: pd.DataFrame, customers: np.ndarray, target_week: int):
    return _generate_candidates_lstm(transactions, customers, target_week, 'gru4rec')


def make_customers_feature(customers_df: pd.DataFrame, transactions: pd.DataFrame, target_week: int, debug: bool = False):
    '''customer毎に特徴量エンジニアリング'''
    df = transactions.copy()
    customers_feature = customers_df.drop(['postal_code'], axis=1).copy()
    customers_feature.loc[~customers_feature['fashion_news_frequency'].isin(['Regularly', 'Monthly']), 'fashion_news_frequency'] = None  # Noneの表記揃え
    customers_feature[['FN', 'Active']] = customers_feature[['FN', 'Active']].fillna(0)

    # week列のシフト(week=target_week => week=0)、リーク防止
    df['week'] = df['week'] - target_week
    df = df.query('week >= 1')
    if debug == True:
        df = df.query('week <= 24')

    weekly_purchase = df.groupby(['customer_id', 'week'])['week'].count().rename('purchase').reset_index()
    
    # 統計量で特徴量
    for agg_name in ['max', 'min', 'mean', 'sum']:
        agg_sr = weekly_purchase.groupby('customer_id')['purchase'].agg(agg_name)
        customers_feature[f'purchase_{agg_name}_groupby_customer'] = customers_feature['customer_id'].map(agg_sr)
    
    # 各週の購入数、統計量との差、比で特徴量
    for w in df['week'].unique()[::-1]:
        tmp = weekly_purchase[weekly_purchase['week']==w]
        tmp = tmp[['customer_id', 'purchase']].set_index('customer_id')['purchase']
        customers_feature[f'purchase_{w}w'] = customers_feature['customer_id'].map(tmp).fillna(0)
        for agg_name in ['max', 'min', 'mean', 'sum']:
            customers_feature[f'purchase_{agg_name}_groupby_customer_ratio_{w}w'] = customers_feature[f'purchase_{w}w'] / customers_feature[f'purchase_{agg_name}_groupby_customer']
            customers_feature[f'purchase_{agg_name}_groupby_customer_diff_{w}w'] = customers_feature[f'purchase_{w}w'] - customers_feature[f'purchase_{agg_name}_groupby_customer']

    # --- 一意の(article_id, week)を購入単位とみなす ---
    # ※あるarticleを1個買うことを、ふつうは購入単位とみなしている
    # rank: 何回目の購入か
    unique_transactions = df[['customer_id', 'article_id', 'week']].drop_duplicates()
    unique_transactions['rank'] = unique_transactions.groupby(['customer_id', 'article_id'])['week'].rank(method='dense', ascending=False)

    # 再購入したarticleの割合
    customers_feature['repurchase_article'] = customers_feature['customer_id'].map(
        unique_transactions.query('rank >= 2').drop_duplicates(subset=['customer_id', 'article_id']).groupby('customer_id')['article_id'].count()).fillna(0)
    customers_feature['purchase_article'] = customers_feature['customer_id'].map(unique_transactions.drop_duplicates(subset=['customer_id', 'article_id']).groupby('customer_id')['article_id'].count())
    customers_feature['repurchase_article_percent'] = customers_feature['repurchase_article'] / customers_feature['purchase_article']

    # 再購入を含む週の割合
    customers_feature['repurchase_week'] = customers_feature['customer_id'].map(
        unique_transactions.query('rank >= 2').drop_duplicates(subset=['customer_id', 'week']).groupby('customer_id')['week'].count()).fillna(0)
    customers_feature['purchase_week'] = customers_feature['customer_id'].map(
        unique_transactions.drop_duplicates(subset=['customer_id', 'week']).groupby('customer_id')['week'].count())
    customers_feature['repurchase_week_percent'] = customers_feature['repurchase_week'] / customers_feature['purchase_week']
    
    # 再購入の割合
    customers_feature['repurchase_article_and_week'] = customers_feature['customer_id'].map(
        unique_transactions.query('rank >= 2').groupby('customer_id')['customer_id'].count()).fillna(0)
    customers_feature['purchase_article_and_week'] = customers_feature['customer_id'].map(
        unique_transactions.groupby('customer_id')['customer_id'].count())
    customers_feature['repurchase_article_and_week_percent'] = customers_feature['repurchase_article_and_week'] / customers_feature['purchase_article_and_week']
    # --- おわり ---
    
    return customers_feature


def make_articles_feature(articles: pd.DataFrame, transactions: pd.DataFrame, target_week: int, debug: bool = False):
    '''article毎に特徴量エンジニアリング'''
    df = transactions.copy()
    articles_feature = articles.drop(
        ['prod_name', 'product_type_name', 'graphical_appearance_name', 'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name', 'index_name', 'index_group_name', 'section_name', 'garment_group_name', 'prod_name', 'department_name', 'detail_desc'], 
        axis=1).copy()
    
    # week列のシフト(week=target_week => week=0)、リーク防止
    df['week'] = df['week'] - target_week
    df = df.query('week >= 1')
    
    if debug == True:
        df = df.query('week <= 24')

    weekly_sale = df.groupby(['article_id', 'week'])['week'].count().rename('sale').reset_index()
    
    # 統計量で特徴量
    for agg_name in ['max', 'min', 'mean', 'sum']:
        agg_sr = weekly_sale.groupby('article_id')['sale'].agg(agg_name)
        articles_feature[f'sale_{agg_name}_groupby_article'] = articles_feature['article_id'].map(agg_sr)

    # 各週の販売数、統計量との差、比で特徴量
    for w in df['week'].unique()[::-1]:
        tmp = weekly_sale[weekly_sale['week']==w]
        tmp = tmp[['article_id', 'sale']].set_index('article_id')['sale']
        articles_feature[f'sale_{w}w'] = articles_feature['article_id'].map(tmp).fillna(0)
        for agg_name in ['max', 'min', 'mean', 'sum']:
            articles_feature[f'sale_{agg_name}_groupby_article_ratio_{w}w'] = articles_feature[f'sale_{w}w'] / articles_feature[f'sale_{agg_name}_groupby_article']
            articles_feature[f'sale_{agg_name}_groupby_article_diff_{w}w'] = articles_feature[f'sale_{w}w'] - articles_feature[f'sale_{agg_name}_groupby_article']

    # --- 一意の(customer_id, week)を販売単位とみなす ---
    # ※あるcustomerに1つ売れることを、ふつうは販売単位とみなしている
    # rank: 何回目の販売か
    unique_transactions = df[['article_id', 'customer_id', 'week']].drop_duplicates()
    unique_transactions['rank'] = unique_transactions.groupby(['article_id', 'customer_id'])['week'].rank(method='dense', ascending=False)

    # 再販売したcustomerの割合
    articles_feature['resale_customer'] = articles_feature['article_id'].map(
        unique_transactions.query('rank >= 2').drop_duplicates(subset=['article_id', 'customer_id']).groupby('article_id')['customer_id'].count()).fillna(0)
    articles_feature['sale_customer'] = articles_feature['article_id'].map(unique_transactions.drop_duplicates(subset=['article_id', 'customer_id']).groupby('article_id')['customer_id'].count())
    articles_feature['resale_customer_percent'] = articles_feature['resale_customer'] / articles_feature['sale_customer']

    # 再販売を含む週の割合
    articles_feature['resale_week'] = articles_feature['article_id'].map(
        unique_transactions.query('rank >= 2').drop_duplicates(subset=['article_id', 'week']).groupby('article_id')['week'].count()).fillna(0)
    articles_feature['sale_week'] = articles_feature['article_id'].map(
        unique_transactions.drop_duplicates(subset=['article_id', 'week']).groupby('article_id')['week'].count())
    articles_feature['resale_week_percent'] = articles_feature['resale_week'] / articles_feature['sale_week']

    # 再販売の割合
    articles_feature['resale_customer_and_week'] = articles_feature['article_id'].map(
        unique_transactions.query('rank >= 2').groupby('article_id')['article_id'].count()).fillna(0)
    articles_feature['sale_customer_and_week'] = articles_feature['article_id'].map(
        unique_transactions.groupby('article_id')['article_id'].count())
    articles_feature['resale_customer_and_week_percent'] = articles_feature['resale_customer_and_week'] / articles_feature['sale_customer_and_week']
    # --- 終わり ---
    
    return articles_feature


# %%


def make_data_df(transactions: pd.DataFrame, week: int, is_labeled: bool, use_customers: np.ndarray = None, metric_verbose: bool = True, compress_verbose: bool = False):
    '''ランク学習モデルへの入力データ作成'''
    # WARNING: 戦略を追加した時に書き換え忘れ注意！
    strategy_flags = [
        'isin_trending', 
        'isin_recently', 
        # 'isin_popular', 
        'isin_bin_popular', 
        'isin_lastw', 
        'isin_pair', 
        'isin_collabo', 
        'isin_tsne_knn', 
        'isin_albert_knn',
        # 'isin_bert_knn',
        # 'isin_vgg_knn',
        'isin_sequence_word2vec_knn',
        # 'isin_lstm',
    ]
    kwargs = {'how': 'outer', 'on': ['customer_id', 'article_id'], 'copy': True}
    
    # 入力データに含むcustomersの絞り込み
    if use_customers is not None:
        data_customers = use_customers
    elif week >= 0:
        data_customers = transactions.query("week == @week")['customer_id'].unique()
    else:
        raise ValueError('set use_customers as something when week=-1.')
    
    # 候補作り
    data_dfs = []
    data_dfs.append(generate_candidates_trending(transactions, data_customers, week))
    data_dfs.append(generate_candidates_recently(transactions, data_customers, week))
    data_dfs.append(generate_candidates_popular(transactions, data_customers, week))
    data_dfs.append(generate_candidates_bin_popular(transactions, data_customers, week))
    data_dfs.append(generate_candidates_lastw(transactions, data_customers, week))
    data_dfs.append(generate_candidates_pair(transactions, data_customers, week))
    data_dfs.append(generate_candidates_collabo(transactions, data_customers, week))
    data_dfs.append(generate_candidates_desc_knn(transactions, data_customers, week))
    # data_dfs.append(generate_candidates_im_knn(transactions, data_customers, week))
    data_dfs.append(generate_candidates_sequence_word2vec(transactions, data_customers, week))
    # data_dfs.append(generate_candidates_lstm(transactions, data_customers, week))
    
    data_df = reduce(
        lambda  left,right: pd.merge(left, right, on=['customer_id', 'article_id'], how='outer'), 
        data_dfs)

    data_df[strategy_flags] = data_df[strategy_flags].fillna(0)
    
    # 正解ラベル付け
    if is_labeled:
        if week < 0:
            raise ValueError(f"can't label when week={week}.")
        data_actual = transactions.query("week == @week")[['customer_id', 'article_id']].drop_duplicates()
        data_actual['label'] = 1
        data_df = data_df.merge(data_actual, how='left', on=['customer_id', 'article_id'])
        data_df['label'] = data_df['label'].fillna(0)
    
    # customers, articles の特徴量
    data_df = compress_df(data_df, verbose=compress_verbose)
    data_customers_feature = compress_df(
        make_customers_feature(customers, transactions, target_week=week, debug=True), 
        verbose=compress_verbose)
    data_articles_feature = compress_df(
        make_articles_feature(articles, transactions, target_week=week, debug=True), 
        verbose=compress_verbose)
    data_df = data_df.merge(data_customers_feature, how='left', on=['customer_id'])
    data_df = data_df.merge(data_articles_feature, how='left', on=['article_id'])
    # data_df = compress_df(data_df, verbose=compress_verbose)
    
    # 候補のスコア
    if metric_verbose:
        print(f"[Info] shape     : {data_df.shape}")
        print(f"[Info] mem       : {data_df.memory_usage().sum() / 1024**2 :5.2f} Mb")
        print(f"[Info] candidates: {len(data_df) / len(data_customers):.1f} 個 / customer")
        print(data_df[strategy_flags].astype(float).mean())
        if is_labeled:
            print(f"[Info] Precision: {data_df['label'].sum() / len(data_df):.5f}")
            print(f"[Info] Recall   : {data_df['label'].sum() / len(data_actual):.5f}")

    return data_df


# %%


def compress_df(
    df: pd.DataFrame, 
    category_columns: list =['club_member_status', 'fashion_news_frequency', 'product_group_name', 'index_code'], 
    verbose: bool =True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    '''DataFrameのデータ型を適切に選び圧縮'''
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        bar = tqdm(df.columns, leave=False)
    else:
        bar = df.columns
    for col in bar:
        col_type = df[col].dtypes
        if col in category_columns:
            if verbose:
                bar.set_description(f"{col}(category)")
            df[col] = df[col].astype('category')
        elif col_type in numerics:
            if verbose:
                bar.set_description(f"{col}(num)")
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# %%


def train(params: dict, cols: list, tr_df: pd.DataFrame, val_df: pd.DataFrame = None, early_stopping: bool = True):
    if val_df is None:
        tr_df = tr_df.sort_values('customer_id').reset_index(drop=True)
        train_query = tr_df.groupby('customer_id')['customer_id'].count().to_list()
        dtrain = lgb.Dataset(tr_df[cols], label=tr_df['label'], group=train_query)
        model = lgb.train(params, dtrain, valid_sets=[dtrain], callbacks=[lgb.log_evaluation(50)])

    else:
        tr_df = tr_df.sort_values('customer_id').reset_index(drop=True)
        val_df = val_df.sort_values('customer_id').reset_index(drop=True)    
        train_query = tr_df.groupby('customer_id')['customer_id'].count().to_list()
        val_query = val_df.groupby('customer_id')['customer_id'].count().to_list()
        dtrain = lgb.Dataset(tr_df[cols], label=tr_df['label'], group=train_query)
        dval = lgb.Dataset(val_df[cols], reference=dtrain, label=val_df['label'], group=val_query)
        if early_stopping:
            model = lgb.train(params, dtrain, valid_sets=[dtrain, dval], callbacks=[lgb.early_stopping(50, first_metric_only=True), lgb.log_evaluation(50)])
        else:
            model = lgb.train(params, dtrain, valid_sets=[dtrain, dval], callbacks=[lgb.log_evaluation(50)])

    return model


# %%


def predict(test_df: pd.DataFrame, model_folds: list):
    pred = np.zeros(len(test_df))
    for w in model_folds:
        with open(f"../models/lgb_rank/{EXP}_model_fold{w}.pkl", 'rb') as f:
            model = pickle.load(f)
        pred += model.predict(test_df[cols], num_iteration=model.best_iteration)    
    pred = pred/len(model_folds)
    
    return pred


# %%


def extract_top_sr(test_df: pd.DataFrame, pred: np.ndarray):
    '''モデルのスコアを用いて並び替え＆上位12個抽出'''
    test_df['predict_score'] = pred
    test_df = test_df.sort_values('predict_score', ascending=False).drop_duplicates(['customer_id', 'article_id'], keep='first').reset_index(drop=True)
    test_df['rank'] = test_df.groupby('customer_id')['predict_score'].rank('min', ascending=False)
    test_df = test_df[test_df['rank'] <= 12]
    
    # test_df['article_id'] = le.inverse_transform(test_df['article_id'])
    test_df['article_id'] = ' 0' + test_df['article_id'].astype(str)
    
    top_sr = test_df.groupby('customer_id')['article_id'].sum()
    
    return top_sr


# # %%


# # optunaによるチューニング
# import optuna.integration.lightgbm as lgb_optuna
# from optuna.logging import set_verbosity
# import numpy as np
# import random as rn

# set_verbosity(-1)
# np.random.seed(71)
# rn.seed(71)

# params = {
#     'objective': 'lambdarank',
#     'boosting': 'gbdt',  # default: 'gbdt', 'gbdt' or 'dart'
#     'num_iterations': 1000,
#     'learning_rate': 0.1,
#     'num_threads': cpu_count(logical=False),
#     'metric': ['ndcg'],
#     'eval_at': [12],  # 上位何件のランキングをnDCGとMAPの算出に用いるか
#     'random_state': 71,  # 訓練用とは違う値にする
#     'verbosity': -1,  # -1: ignore, 0: warnings, 1: info
#     'deterministic': True,  # 再現性確保
#     'force_col_wise': True  # 再現性確保
# }

# tr_df = make_data_df(transactions, 2, is_labeled=True, metric_verbose=False, compress_verbose=False)
# exclude_columns = ['target_week', 'customer_id', 'article_id', 'label']
# cols = [c for c in tr_df.columns.tolist() if c not in exclude_columns]
# tr_df = tr_df.sort_values('customer_id').reset_index(drop=True)
# train_query = tr_df.groupby('customer_id')['customer_id'].count().to_list()
# dtrain = lgb.Dataset(tr_df[cols], label=tr_df['label'], group=train_query)

# val_df = make_data_df(transactions, 1, is_labeled=True, metric_verbose=False, compress_verbose=False)
# val_df = val_df.sort_values('customer_id').reset_index(drop=True)    
# val_query = val_df.groupby('customer_id')['customer_id'].count().to_list()
# dval = lgb.Dataset(val_df[cols], reference=dtrain, label=val_df['label'], group=val_query)

# model = lgb_optuna.LightGBMTuner(params, dtrain, valid_sets=[dtrain, dval], early_stopping_rounds=10, verbose_eval=-1, optuna_seed=71)
# model.run()

# best_params = model.params
# with open(f'../models/lgb_rank/{EXP}_best_params.pkl', 'wb') as f:
#     pickle.dump(best_params, f)


# %%


# ランク学習
params = {
    # Core Parameters
    'objective': 'lambdarank',
    'boosting': 'gbdt',  # default: 'gbdt', ['gbdt', 'dart', 'goss'], dart: 超遅いけど高精度
    'num_iterations': 1000,
    'learning_rate': 0.02,
    'num_leaves': 31,  # default: 31, large for accuracy
    'num_threads': cpu_count(logical=False),
    'random_state': 41,

    # Learning Control Parameters
    'force_col_wise': True,
    # 'histogram_pool_size': -1.0,  # default: -1.0, max cache size in MB for historical histogram, (histogram_pool_size + dataset size) = approximately RAM used
    'min_data_in_leaf': 20,  # default: 20, large dealing with over-fitting
    'min_sum_hessian_in_leaf': 1e-3,  # default: 1e-3, large dealing with over-fitting
    'max_depth': -1,
    'bagging_freq': 5,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.8,
    # 'drop_rate': 0.1,  # used only in dart, a fraction of previous trees to drop during the dropout
    'verbosity': 0,  # 0: warnings, 1: info

    # Dataset Parameters
    'max_bin': 255,  # default: 255, large for accuracy
    'min_data_in_bin': 3,  # default: 3, 
    'bin_construct_sample_cnt': 200000,  # default: 200000, larger for accuracy

    # Objective Parameters
    'lambdarank_truncation_level': 30,
    'lambdarank_norm': True,
    # 'label_gain': [0, 1],

    # Metric Parameters
    'metric': ['ndcg'],
    'eval_at': [12],  # 上位何件のランキングをnDCGとMAPの算出に用いるか
}


# %%


# train lgb ranker
best_iterations = []
feature_importance_dfs = []
oof_weeks = [4, 3, 2, 1]

# for i, w in enumerate(tqdm(oof_weeks)):
#     print(f"\ntarget_week(fold): {w}")
#     if i == 0:
#         compress_verbose=True
#     else:
#         compress_verbose=False

#     tr_df = make_data_df(transactions, w, is_labeled=True, metric_verbose=True, compress_verbose=compress_verbose)
#     val_df = make_data_df(transactions, w-1, is_labeled=True, metric_verbose=False, compress_verbose=False)

#     if i == 0:
#         exclude_columns = ['target_week', 'customer_id', 'article_id', 'label']
#         cols = [c for c in tr_df.columns.tolist() if c not in exclude_columns]
#         with open(f'../models/lgb_rank/{EXP}_cols.pkl', 'wb') as f:
#             pickle.dump(cols, f)

#     model = train(params, cols, tr_df, val_df)
#     with open(f'../models/lgb_rank/{EXP}_model_fold{w}.pkl', 'wb') as f:
#         pickle.dump(model, f)

#     best_iterations.append(model.best_iteration)
#     feature_importance_dfs.append(pd.DataFrame({'feature': model.feature_name(), 'importance(gain)': model.feature_importance('gain'), 'fold': w}))

# send_line_notification(f'[EXP{EXP}]\ntrain finished exceped for last week.')


# # %%


# # predict val data
# val_df = make_data_df(transactions, 0, is_labeled=False, metric_verbose=True, compress_verbose=False)
# val_pred = predict(val_df, oof_weeks)
# print(np.sort(val_pred))


# # %%


# # val top rank articles
# val_df2 = val_df.copy()
# val_pred_sr = extract_top_sr(val_df2, val_pred)
# print(val_pred_sr.head(3))


# # %%


# # most popular items
# transactions_last_week = transactions.loc[transactions.week == 1]
# top12 = ' 0' + ' 0'.join(transactions_last_week.article_id.value_counts().index.astype('str')[:12])

# customers2 = customers.copy()
# customers2['age_bin'] = pd.cut(customers2['age'], bins=[10, 20, 30, 40, 50, 60, 70, 100], labels=False)
# transactions_last_week = transactions_last_week.merge(customers2[['customer_id', 'age', 'age_bin']], how='left')
# popular_items = transactions_last_week.groupby('age_bin')['article_id'].value_counts()
# popular_items_dict = {}
# for index in popular_items.index.levels[0]:
#     popular_items_dict[index] = ' 0'+' 0'.join(popular_items[index][:12].index.astype('str'))
# popular_items_sr = pd.Series(popular_items_dict, name='top_12_popular_items', dtype='str')


# # %%


# # val sub
# submission = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv')

# submission['prediction_lgb'] = submission['customer_id'].map(id_to_index_dict).map(val_pred_sr)
# submission['prediction_lgb'] = submission['prediction_lgb'].fillna('')

# submission['age_bin'] = submission['customer_id'].map(id_to_index_dict).map(customers2.set_index('customer_id')['age_bin'])
# submission['prediction_popular'] = submission['age_bin'].map(popular_items_sr)
# submission['prediction_popular'] = submission['prediction_popular'].fillna(top12).astype('str')

# submission['prediction'] = submission['prediction_lgb'] + submission['prediction_popular']
# submission['prediction'] = submission['prediction'].str.strip()
# submission['prediction'] = submission['prediction'].str[:131]
# print(submission.head(3))
# submission[['customer_id', 'prediction']].to_csv(f'../submissions/{EXP}_submission_fold0.csv', index=False)


# # %%


# del val_df, model
# del transactions_last_week, top12, customers2, popular_items, popular_items_dict, popular_items_sr
# del val_df2, val_pred_sr, 
# del submission
# gc.collect()


# # %%


# # train last target_week(=0) data
# tr_df = make_data_df(transactions, week=0, is_labeled=True, metric_verbose=True, compress_verbose=False)

# # valがないのでアーリーストッピングが使えない
# params['num_iterations'] = int(np.mean(best_iterations))
# model = train(params, cols, tr_df)
# with open(f"../models/lgb_rank/{EXP}_model_fold0.pkl", 'wb') as f:
#     pickle.dump(model, f)

# feature_importance_dfs.append(pd.DataFrame({'feature': model.feature_name(), 'importance(gain)': model.feature_importance('gain'), 'fold': 0}))


# # %%


# feature_importance_df = pd.concat(feature_importance_dfs, ignore_index=True, axis=0)
# print(feature_importance_df.groupby(['feature'])[['importance(gain)']].mean().sort_values('importance(gain)', ascending=False).head(20))


# # %%


# del tr_df, params, model, best_iterations, feature_importance_dfs
# gc.collect()


# %%


with open(f'../models/lgb_rank/{EXP}_cols.pkl', 'rb') as f:
    cols = pickle.load(f)
# predict test data
BATCH_SIZE = 200_000
submission = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv')
test_customers = submission['customer_id'].map(id_to_index_dict).unique()

# バッチ処理
def process(i):
    if i == (len(test_customers)//BATCH_SIZE):
        test_customers_batch = test_customers[i*BATCH_SIZE : ]
    else:
        test_customers_batch = test_customers[i*BATCH_SIZE : (i+1)*BATCH_SIZE]

    if i == 0:  # meric_verbose=True
        test_df = make_data_df(transactions, week=-1, is_labeled=False, use_customers=test_customers_batch, metric_verbose=True, compress_verbose=False)
    else:       # metric_verbose=False
        test_df = make_data_df(transactions, week=-1, is_labeled=False, use_customers=test_customers_batch, metric_verbose=False, compress_verbose=False)

    all_weeks = oof_weeks + [0]
    pred = predict(test_df, all_weeks)
    return extract_top_sr(test_df, pred)

# single process execution
preds = []
for i in tqdm(range(len(test_customers)//BATCH_SIZE + 1)):
    preds.append(process(i))

# # multi process execution
# # cpus = cpu_count(logical=False)
# cpus = 4
# print('cpu(core): ', cpus)
# preds = Parallel(n_jobs=cpus, verbose=0)( [delayed(process)(i) for i in range(len(test_customers)//BATCH_SIZE + 1)] )

pred_sr = pd.concat(preds, axis=0)
print(pred_sr.head(3))


# %%


del preds
gc.collect()


# %%


# # most popular items
# transactions_last_week = transactions.loc[transactions.week == 0]
# top12 = ' 0' + ' 0'.join(transactions_last_week.article_id.value_counts().index.astype('str')[:12])
# print("Top 12 popular items:")
# print( top12 )

# customers['age_bin'] = pd.cut(customers['age'], bins=[10, 20, 30, 40, 50, 60, 70, 100], labels=False)
# transactions_last_week = transactions_last_week.merge(customers[['customer_id', 'age', 'age_bin']], how='left')
# popular_items = transactions_last_week.groupby('age_bin')['article_id'].value_counts()
# popular_items_dict = {}
# for index in popular_items.index.levels[0]:
#     popular_items_dict[index] = ' 0'+' 0'.join(popular_items[index][:12].index.astype('str'))
# popular_items_sr = pd.Series(popular_items_dict, name='top_12_popular_items', dtype='str')
# popular_items_sr


# %%


# test sub
submission = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv')

submission['prediction_lgb'] = submission['customer_id'].map(id_to_index_dict).map(pred_sr)
submission['prediction_lgb'] = submission['prediction_lgb'].fillna('')

# submission['age_bin'] = submission['customer_id'].map(id_to_index_dict).map(customers.set_index('customer_id')['age_bin'])
# submission['prediction_popular'] = submission['age_bin'].map(popular_items_sr)
# submission['prediction_popular'] = submission['prediction_popular'].fillna(top12).astype('str')

submission['prediction'] = submission['prediction_lgb']
submission['prediction'] = submission['prediction'].str.strip()
submission['prediction'] = submission['prediction'].str[:131]
print(submission.head(3))
submission[['customer_id', 'prediction']].to_csv(f'../submissions/{EXP}_submission.csv', index=False)

send_line_notification(f'[EXP{EXP}]\nprediction finished.')
