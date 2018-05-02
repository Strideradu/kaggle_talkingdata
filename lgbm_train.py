"""
A non-blending lightGBM model that incorporates portions and ideas from various public kernels
This kernel gives LB: 0.977 when the parameter 'debug' below is set to 0 but this implementation requires a machine with ~32 GB of memory
"""

import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc
import matplotlib
import argparse
import sys

matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

debug = 0

nrows = 184903891 - 1
nchunk = 40000000
val_size = 4000000

# frm = nrows - 75000000
frm = 0
nchunk = nrows - 1
if debug:
    frm = 0
    nchunk = 100000
    val_size = 10000

to = frm + nchunk

if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')


def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                      feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10,
                      categorical_features=None, test = None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric': metrics,
        'learning_rate': 0.2,
        # 'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 28,
        'verbose': 0,
        'metric': metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )




    if test:
        bst1 = lgb.train(lgb_params,
                         xgtrain,
                         num_boost_round=test,
                         verbose_eval=10,
                         feval=feval)
        return bst1, test
    else:
        evals_results = {}
        xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                              feature_name=predictors,
                              categorical_feature=categorical_features
                              )
        bst1 = lgb.train(lgb_params,
                         xgtrain,
                         valid_sets=[xgtrain, xgvalid],
                         valid_names=['train', 'valid'],
                         evals_result=evals_results,
                         num_boost_round=num_boost_round,
                         early_stopping_rounds=early_stopping_rounds,
                         verbose_eval=10,
                         feval=feval)

        print("\nModel Report")
        print("bst1.best_iteration: ", bst1.best_iteration)
        print(metrics + ":", evals_results['valid'][metrics][bst1.best_iteration - 1])

        return (bst1, bst1.best_iteration)


def DO(frm, to, fileno, test = None):
    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'uint32',
    }

    print('loading train data...', frm, to)
    if test:
        # use 8, 9 prdict test
        train_df = pd.read_csv("/mnt/home/dunan/Learn/Kaggle/talkingdata_fraud/train.csv", parse_dates=['click_time'],
                               skiprows=range(1, 68941879), nrows=to - 68941879,
                               dtype=dtypes,
                               usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])

        print('loading test data...')
        test_df = pd.read_csv("/mnt/home/dunan/Learn/Kaggle/talkingdata_fraud/test.csv", parse_dates=['click_time'],
                              dtype=dtypes,
                              usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])

        len_train = len(train_df)
        len_test = len(test_df)
        train_df = train_df.append(test_df)

        del test_df
        gc.collect()
    else:
        # use 7, 8 to predict on 9
        train_df = pd.read_csv("/mnt/home/dunan/Learn/Kaggle/talkingdata_fraud/train.csv", parse_dates=['click_time'],
                               skiprows=range(1, 9308569), nrows=to - 9308569,
                               dtype=dtypes,
                               usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])
        len_train = len(train_df)



    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    train_df['wday'] = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8')

    gc.collect()

    naddfeat = 9
    for i in range(0, naddfeat):
        if i == 0: selcols = ['ip', 'channel']; QQ = 4;
        if i == 1: selcols = ['ip', 'device', 'os', 'app']; QQ = 5;
        if i == 2: selcols = ['ip', 'day', 'hour']; QQ = 4;
        if i == 3: selcols = ['ip', 'app']; QQ = 4;
        if i == 4: selcols = ['ip', 'app', 'os']; QQ = 4;
        if i == 5: selcols = ['ip', 'device']; QQ = 4;
        if i == 6: selcols = ['app', 'channel']; QQ = 4;
        if i == 7: selcols = ['ip', 'os']; QQ = 5;
        if i == 8: selcols = ['ip', 'device', 'os', 'app']; QQ = 4;
        print('selcols', selcols, 'QQ', QQ)

        filename = 'X%d_%d_%d.csv' % (i, frm, to)

        if os.path.exists(filename):
            if QQ == 5:
                gp = pd.read_csv(filename, header=None)
                train_df['X' + str(i)] = gp
            else:
                gp = pd.read_csv(filename)
                train_df = train_df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
        else:
            if QQ == 0:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols) - 1])[
                    selcols[len(selcols) - 1]].count().reset_index(). \
                    rename(index=str, columns={selcols[len(selcols) - 1]: 'X' + str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
            if QQ == 1:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols) - 1])[
                    selcols[len(selcols) - 1]].mean().reset_index(). \
                    rename(index=str, columns={selcols[len(selcols) - 1]: 'X' + str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
            if QQ == 2:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols) - 1])[
                    selcols[len(selcols) - 1]].var().reset_index(). \
                    rename(index=str, columns={selcols[len(selcols) - 1]: 'X' + str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
            if QQ == 3:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols) - 1])[
                    selcols[len(selcols) - 1]].skew().reset_index(). \
                    rename(index=str, columns={selcols[len(selcols) - 1]: 'X' + str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
            if QQ == 4:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols) - 1])[
                    selcols[len(selcols) - 1]].nunique().reset_index(). \
                    rename(index=str, columns={selcols[len(selcols) - 1]: 'X' + str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols) - 1], how='left')
            if QQ == 5:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols) - 1])[selcols[len(selcols) - 1]].cumcount()
                train_df['X' + str(i)] = gp.values

            if not debug:
                gp.to_csv(filename, index=False)

        del gp
        gc.collect()

    print('doing nextClick')
    predictors = []

    new_feature = 'nextClick'
    filename = 'nextClick_%d_%d.csv' % (frm, to)

    if os.path.exists(filename):
        print('loading from save file')
        if test:
            QQ = pd.read_csv(filename)[68941879:].values
        else:
            QQ = pd.read_csv(filename)[9308569:9308569 + len_train].values
    else:
        D = 2 ** 26
        train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df[
            'device'].astype(str) \
                                + "_" + train_df['os'].astype(str)).apply(hash) % D
        click_buffer = np.full(D, 3000000000, dtype=np.uint32)

        train_df['epochtime'] = train_df['click_time'].astype(np.int64) // 10 ** 9
        next_clicks = []
        for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
            next_clicks.append(click_buffer[category] - t)
            click_buffer[category] = t
        del (click_buffer)
        QQ = list(reversed(next_clicks))

        if not debug:
            print('saving')
            pd.DataFrame(QQ).to_csv(filename, index=False)

    # also add prev click?

    train_df[new_feature] = QQ
    predictors.append(new_feature)

    train_df[new_feature + '_shift'] = pd.DataFrame(QQ).shift(+1).values
    predictors.append(new_feature + '_shift')

    del QQ
    gc.collect()

    print('grouping by ip-day-hour combination...')
    gp = train_df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
    train_df = train_df.merge(gp, on=['ip', 'day', 'hour'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app combination...')
    gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(
        index=str, columns={'channel': 'ip_app_count'})
    train_df = train_df.merge(gp, on=['ip', 'app'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app-os combination...')
    gp = train_df[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    train_df = train_df.merge(gp, on=['ip', 'app', 'os'], how='left')
    del gp
    gc.collect()

    # new fatures

    most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
    least_freq_hours_in_test_data = [6, 11, 15]

    train_df['in_test_hh'] = (3
                              - 2 * train_df['hour'].isin(most_freq_hours_in_test_data)
                              - 1 * train_df['hour'].isin(least_freq_hours_in_test_data)).astype('uint8')

    # unique count

    # chennel

    print('grouping by ip-app-os combination...')
    gp = train_df[['app', 'channel']].groupby(by=['app'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'app_count'})
    train_df = train_df.merge(gp, on=['app'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app-os combination...')
    gp = train_df[['os', 'app', 'channel']].groupby(by=['os', 'app'])[
        ['channel']].count().reset_index().rename(index=str, columns={'channel': 'os_app_count'})
    train_df = train_df.merge(gp, on=['os', 'app'], how='left')
    del gp
    gc.collect()

    print('grouping by ip combination...')
    gp = train_df[['ip', 'device']].groupby(by=['ip'])[['device']].nunique().reset_index().rename(
        index=str, columns={'device': 'ip_unique_device'})
    train_df = train_df.merge(gp, on=['ip'], how='left')
    del gp
    gc.collect()

    print('grouping by os combination...')
    gp = train_df[['os', 'device']].groupby(by=['os'])[['device']].nunique().reset_index().rename(
        index=str, columns={'device': 'os_unique_device'})
    train_df = train_df.merge(gp, on=['os'], how='left')
    del gp
    gc.collect()

    print('grouping by ip combination...')
    gp = train_df[['ip', 'app']].groupby(by=['ip'])[['app']].nunique().reset_index().rename(
        index=str, columns={'app': 'ip_unique_app'})
    train_df = train_df.merge(gp, on=['ip'], how='left')
    del gp
    gc.collect()


    print('grouping by ip-device-os combination...')
    gp = train_df[['ip', 'device', 'os', 'app']].groupby(by=['ip', 'device', 'os'])[
        ['app']].nunique().reset_index().rename(
        index=str, columns={'app': 'ip_device_os_unique_app'})
    train_df = train_df.merge(gp, on=['ip', 'device', 'os'], how='left')
    del gp
    gc.collect()


    print('grouping by wday-hour combination...')
    gp = train_df[['wday', 'hour', 'app']].groupby(by=['wday', 'hour'])[
        ['app']].nunique().reset_index().rename(
        index=str, columns={'app': 'wday_hour_unique_app'})
    train_df = train_df.merge(gp, on=['wday', 'hour'], how='left')
    del gp
    gc.collect()


    print('grouping by ip-app combination...')
    gp = train_df[['ip', 'app', 'os']].groupby(by=['ip', 'app'])[['os']].nunique().reset_index().rename(
        index=str, columns={'os': 'ip_app_unique_os'})
    train_df = train_df.merge(gp, on=['ip', 'app'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-wday combination...')
    gp = train_df[['ip', 'wday', 'in_test_hh']].groupby(by=['ip', 'wday'])[
        ['in_test_hh']].nunique().reset_index().rename(
        index=str, columns={'in_test_hh': 'ip_wday_unique_in_test_hh'})
    train_df = train_df.merge(gp, on=['ip', 'wday'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-day combination...')
    gp = train_df[['ip', 'day', 'hour']].groupby(by=['ip', 'day'])[['hour']].nunique().reset_index().rename(
        index=str, columns={'hour': 'ip_day_unique_hour'})
    train_df = train_df.merge(gp, on=['ip', 'day'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-wday combination...')
    gp = train_df[['ip', 'wday', 'hour']].groupby(by=['ip', 'wday'])[['hour']].nunique().reset_index().rename(
        index=str, columns={'hour': 'ip_wday_unique_hour'})
    train_df = train_df.merge(gp, on=['ip', 'wday'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-os-wday combination...')
    gp = train_df[['ip', 'os', 'wday', 'hour']].groupby(by=['ip', 'os', 'wday'])[
        ['hour']].nunique().reset_index().rename(
        index=str, columns={'hour': 'ip_os_wday_unique_hour'})
    train_df = train_df.merge(gp, on=['ip', 'os', 'wday'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app-wday combination...')
    gp = train_df[['ip', 'app', 'wday', 'hour']].groupby(by=['ip', 'app', 'wday'])[
        ['hour']].nunique().reset_index().rename(
        index=str, columns={'hour': 'ip_app_wday_unique_hour'})
    train_df = train_df.merge(gp, on=['ip', 'app', 'wday'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-device-wday combination...')
    gp = train_df[['ip', 'device', 'wday', 'hour']].groupby(by=['ip', 'device', 'wday'])[
        ['hour']].nunique().reset_index().rename(
        index=str, columns={'hour': 'ip_device_wday_unique_hour'})
    train_df = train_df.merge(gp, on=['ip', 'device', 'wday'], how='left')
    del gp
    gc.collect()


    # Adding features with var and mean hour (inspired from nuhsikander's script)
    print('grouping by : ip_day_chl_var_hour')
    gp = train_df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'channel'])[
        ['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
    train_df = train_df.merge(gp, on=['ip', 'day', 'channel'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_os_var_hour')
    gp = train_df[['ip', 'app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(
        index=str, columns={'hour': 'ip_app_os_var'})
    train_df = train_df.merge(gp, on=['ip', 'app', 'os'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_channel_var_day')
    gp = train_df[['ip', 'app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[
        ['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
    train_df = train_df.merge(gp, on=['ip', 'app', 'channel'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_chl_mean_hour')
    gp = train_df[['ip', 'app', 'channel', 'hour']].groupby(by=['ip', 'app', 'channel'])[
        ['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
    print("merging...")
    train_df = train_df.merge(gp, on=['ip', 'app', 'channel'], how='left')
    del gp
    gc.collect()

    # cumcount feature

    print('grouping by ip combination...')
    gp = train_df[['ip', 'os']].groupby(by=['ip'])[['os']].cumcount()
    train_df['ip_cum_os'] = gp.values
    del gp
    gc.collect()

    print('grouping by ip-device-os combination...')
    gp = train_df[['ip', 'device', 'os', 'app']].groupby(by=['ip', 'device', 'os'])[['app']].cumcount()
    train_df['ip_device_os_cum_app'] = gp.values
    del gp
    gc.collect()

    print("vars and data type: ")
    train_df.info()
    train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
    train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

    target = 'is_attributed'
    predictors.extend(['app', 'device', 'os', 'channel', 'hour', 'day',
                       'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                       'ip_app_os_count', 'ip_app_os_var',
                       'ip_app_channel_var_day', 'ip_app_channel_mean_hour',
                       'in_test_hh', 'ip_wday_unique_in_test_hh', 'ip_device_os_unique_app', 'ip_unique_device',
                       'ip_unique_app', 'ip_app_unique_os', 'ip_day_unique_hour',
                       'ip_cum_os', 'ip_device_os_cum_app', 'ip_wday_unique_hour',
                       'ip_os_wday_unique_hour', 'ip_app_wday_unique_hour', 'ip_device_wday_unique_hour',
                       'os_app_count', 'app_count', 'os_unique_device', 'wday_hour_unique_app'
                       ])
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
    for i in range(0, naddfeat):
        predictors.append('X' + str(i))

    print('predictors', predictors)

    if test:
        test_df = train_df[len_train:]
        train_df = train_df[:len_train ]
        val_df = None
    else:
        test_df = None
        val_size = to - 131886954
        val_df = train_df[(len_train - val_size):len_train]
        train_df = train_df[:(len_train - val_size)]


    print("train size: ", len(train_df))

    if test:
        sub = pd.DataFrame()
        sub['click_id'] = test_df['click_id'].astype('int')

    gc.collect()

    print("Training...")
    start_time = time.time()

    params = {
        'learning_rate': 0.20,
        # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight': 200  # because training data is extremely unbalanced
    }

    (bst, best_iteration) = lgb_modelfit_nocv(params,
                                              train_df,
                                              val_df,
                                              predictors,
                                              target,
                                              objective='binary',
                                              metrics='auc',
                                              early_stopping_rounds=30,
                                              verbose_eval=True,
                                              num_boost_round=1000,
                                              categorical_features=categorical,
                                              test = test)

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()

    if test:
        print("Predicting...")
        sub['is_attributed'] = bst.predict(test_df[predictors], num_iteration=best_iteration)
        if not debug:
            print("writing...")
            sub.to_csv('sub_it%d.csv.gz' % (fileno), index=False, compression='gzip')
        print("done...")

        return sub
    else:
        print('Plot feature importances...')
        ax = lgb.plot_importance(bst, max_num_features=100)
        plt.savefig("result.png")

        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="best of num of epoches", type=int, default=None)
    try:
        args = parser.parse_args()

    except:
        parser.print_help()
        sys.exit(1)
    sub = DO(frm, to, 0, test=args.test)
