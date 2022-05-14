# ホストが指定した通りのMAPの実装
# 実際にsubするのと同じフォーマットで、foldの予測作成
# 09-15 < fold0 <= 09-22

import numpy as np
import pandas as pd
import warnings
import argparse
import os

warnings.simplefilter('ignore', pd.core.common.SettingWithCopyWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--exp")
# parser.add_argument("-f", "--file")
parser.add_argument("--fold")
parser.add_argument("--first_execution", action='store_true')
args = parser.parse_args()

EXP = '001' if args.exp is None else args.exp
FOLD = '0' if args.fold is None else args.fold
SUBMISSION_PATH = f'../submissions/{EXP}_submission_fold{FOLD}.csv'  # CSVファイルの指定
VALID_PATH = f'../input/valid/transactions_valid_fold{FOLD}.csv'
CUSTOMER_PATH = f'../input/valid/customers_is_transaction_fold{FOLD}.npy'
VALID_START = pd.to_datetime({'0': '2020-09-15', '1': '2020-09-08', '2': '2020-09-01'}[FOLD])  # excluded
VALID_END = pd.to_datetime({'0': '2020-09-22', '1': '2020-09-15', '2': '2020-09-08'}[FOLD])  # include


def average_precision_at_k(actual, predicted, k=12):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mean_average_precision(y_true, y_pred, k=12):
    """ Computes MAP at k
    
    Parameters
    __________
    y_true: list.
            nested list of correct recommendations (Order doesn't matter)
    y_pred: list
            nested list of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           MAP at k
    """
    return np.mean([average_precision_at_k(gt, pred, k) \
                    for gt, pred in zip(y_true, y_pred)])


def main():
    # バリデーションの期間に絞った購入履歴
    # 初回だけファイル作成
    if args.first_execution:
        transaction = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv', parse_dates=['t_dat'])
        transaction = transaction.query('t_dat > @VALID_START and t_dat <= @VALID_END')
        transaction['article_id'] = transaction['article_id'].map(str)
        transaction.to_csv(VALID_PATH, index=False)
        print(f'Saved: {VALID_PATH}')
    # 2回目以降は再利用
    print(f'Loading: {VALID_PATH}')
    transaction = pd.read_csv(VALID_PATH, dtype={'article_id': str}, parse_dates=['t_dat'])

    # true and pred dataframe
    true = transaction.groupby('customer_id')['article_id'].apply(lambda items: list(set(items))).reset_index().rename(columns={'article_id': 'true'})
    print(f'Loading: {SUBMISSION_PATH}')
    pred = pd.read_csv(SUBMISSION_PATH)
    pred['prediction'] = pred['prediction'].str.split(' ').map(lambda l: [v[1:] for v in l])
    true_and_pred = pd.merge(true, pred, how='left')

    # コールドスタートのユーザのフラグ
    # 初回だけファイル作成
    if args.first_execution:
        transaction = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv', parse_dates=['t_dat'])
        transaction = transaction.query('t_dat <= @VALID_START')
        customers_transaction = transaction['customer_id'].unique()
        np.save(CUSTOMER_PATH, customers_transaction)
        print(f'Saved: {CUSTOMER_PATH}')
    # 2回目以降は再利用
    print(f'Loading: {CUSTOMER_PATH}')
    customers_transaction = np.load(CUSTOMER_PATH, allow_pickle=True)
    true_and_pred['is_transaction'] = true_and_pred['customer_id'].isin(customers_transaction)

    print(f'Validation Customers (all): {len(true_and_pred)}')
    print(f'Validation Customers (cold start): {len(true_and_pred.query("not is_transaction"))}')
    print(f"MAP@12 (all): {mean_average_precision(true_and_pred['true'].to_list(), true_and_pred['prediction'].to_list(), k=12):.6f}")
    print(f"MAP@12 (cold start): {mean_average_precision(true_and_pred.query('not is_transaction')['true'].to_list(), true_and_pred.query('not is_transaction')['prediction'].to_list(), k=12):.6f}")


if __name__ == '__main__':
    main()
