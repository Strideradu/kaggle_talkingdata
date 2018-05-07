# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

path = "/mnt/home/dunan/Learn/Kaggle/talkingdata_fraud/"

# Any results you write to the current directory are saved as output.
print("Reading the data...\n")
df1 = pd.read_csv(path + 'sub-it200102csv')
df2 = pd.read_csv(path + 'sub_2nd_it23.csv')
df3 = pd.read_csv(path + 'sub_newauc_0p9894.csv')

models = {
    'df1': {
        'name': 'test_supplement',
        'score': 98.11,
        'df': df1
    },
    'df2': {
        'name': '9800_kernal',
        'score': 98.00,
        'df': df2
    },
    'df3': {
        'name': 'public_kernal_full_data',
        'score': 98.94,
        'df': df3
    },
}

count_models = len(models)

isa_lg = 0
isa_hm = 0
isa_am = 0
print("Blending...\n")
for df in models.keys():
    isa_lg += np.log(models[df]['df'].is_attributed)
    isa_hm += 1 / (models[df]['df'].is_attributed)
    isa_am += models[df]['df'].is_attributed
isa_lg = np.exp(isa_lg / count_models)
isa_hm = count_models / isa_hm
isa_am = isa_am / count_models

print("Isa log\n")
print(isa_lg[:count_models])
print()
print("Isa harmo\n")
print(isa_hm[:count_models])

sub_am= pd.DataFrame()
sub_am['click_id'] = df1['click_id']
sub_am['is_attributed'] = isa_am

sub_log = pd.DataFrame()
sub_log['click_id'] = df1['click_id']
sub_log['is_attributed'] = isa_lg
sub_log.head()

sub_hm = pd.DataFrame()
sub_hm['click_id'] = df1['click_id']
sub_hm['is_attributed'] = isa_hm
sub_hm.head()

sub_fin = pd.DataFrame()
sub_fin['click_id'] = df1['click_id']
sub_fin['is_attributed'] = (5.5 * isa_lg + 3 * isa_hm + 1.5 * isa_am) / 10

print("Writing...")
# sub_log.to_csv('submission_log2.csv', index=False, float_format='%.9f')
sub_am.to_csv(path + 'submission_am.csv', index=False, float_format='%.9f')
sub_fin.to_csv(path + 'submission_final.csv', index=False, float_format='%.9f')
