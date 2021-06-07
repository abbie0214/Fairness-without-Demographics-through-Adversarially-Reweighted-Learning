# dataloader

from __future__ import division
import pandas as pd
import numpy as np
import json
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch

import csv
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
# compas dataset
from torch import nn
import config

args = config.parse_args()

###################################################################
"""""
preprocessing for compass dataset

"""""

pd.options.display.float_format = '{:,.2f}'.format
dataset_base_dir = './data/compas/'
dataset_file_name = 'compas-scores-two-years.csv'

# Processing original dataset

file_path = os.path.join(dataset_base_dir, dataset_file_name)
with open(file_path, "r") as file_name:
    temp_df = pd.read_csv(file_name)
print('temp_df', temp_df)
# Columns of interest
columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
           'age',
           'c_charge_degree',
           'c_charge_desc',
           'age_cat',
           'sex', 'race', 'is_recid']
target_variable = 'is_recid'
target_value = 'Yes'

# Drop duplicates
temp_df = temp_df[['id'] + columns].drop_duplicates()
df = temp_df[columns].copy()

# Convert columns of type ``object`` to ``category``
df = pd.concat([df.select_dtypes(include=[], exclude=['object']),
                df.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
                ], axis=1).reindex(df.columns, axis=1)

print('df', df)
# Binarize target_variable
df['is_recid'] = df.apply(lambda x: 'Yes' if x['is_recid'] == 1.0 else 'No', axis=1).astype('category')

# Process protected-column values
race_dict = {'African-American': 'Black', 'Caucasian': 'White'}
df['race'] = df.apply(lambda x: race_dict[x['race']] if x['race'] in race_dict.keys() else 'Other', axis=1).astype(
    'category')

df.head()

# Shuffle and Split into Train (70%) and Test set (30%)

train_df, test_df = train_test_split(df, test_size=0.30, random_state=42)

output_file_path = os.path.join(dataset_base_dir, 'train.csv')
with open(output_file_path, mode="w", newline='') as output_file:
    train_df.to_csv(output_file, index=False, columns=columns, header=False)
    output_file.close()
#
# output_file_path = os.path.join(dataset_base_dir,'test.csv')
# with open(output_file_path, mode="w",newline='') as output_file:
#     test_df.to_csv(output_file,index=False,columns=columns,header=False)
#     output_file.close()


file_path = os.path.join('./data/compas/', 'train_modified.csv')
with open(file_path, "r") as file_name:
    train_df2 = pd.read_csv(file_name)
print('train_df2', train_df2)


###################################################################

# # Construct vocabulary.json, and write to directory:


# This is for showing what are the categorials and the detailed classes in each category.
# For example: {'c_charge_degree': ['M', 'F']}

# cat_cols = train_df.select_dtypes(include='category').columns
# vocab_dict = {}
# for col in cat_cols:
#     vocab_dict[col] = list(set(train_df[col].cat.categories))
#
# output_file_path = os.path.join(dataset_base_dir, 'vocabulary.json')
# with open(output_file_path, mode="w") as output_file:
#     output_file.write(json.dumps(vocab_dict))
#     output_file.close()
# # print('vocab_dict',vocab_dict)
###################################################################
# Construct mean_std.json, and write to directory
# after transfering now 11 all have mean and std
# orginal only 5 has mean and std: because original there are only 5 are enumerical, others are categorial
# temp_dict = train_df2.describe().to_dict()
# mean_std_dict = {}
# for key, value in temp_dict.items():
#   mean_std_dict[key] = [value['mean'],value['std']]
#
# output_file_path = os.path.join('./data/compas/','mean_std.json')
# with open(output_file_path, mode="w") as output_file:
#     output_file.write(json.dumps(mean_std_dict))
#     output_file.close()
# print('mean_std_dict modified',mean_std_dict)
#
# temp_dict = train_df.describe().to_dict()
# mean_std_dict = {}
# for key, value in temp_dict.items():
#   mean_std_dict[key] = [value['mean'],value['std']]
#
# output_file_path = os.path.join('./data/compas/','mean_std_ori.json')
# with open(output_file_path, mode="w") as output_file:
#     output_file.write(json.dumps(mean_std_dict))
#     output_file.close()
# print('mean_std_dict original',mean_std_dict)
###################################################################
class CompasInput():
    """Data reader for Compas dataset."""
    
    def __init__(self,
                 dataset_base_dir,
                 train_file=None,
                 test_file=None):
        """Data reader for Compas dataset.
        Args:
          dataset_base_dir: (string) directory path.
          train_file: string list of training data paths.
          test_file: string list of evaluation data paths.
          dataset_base_sir must contain the following files in the dir:
          - train.csv: comma separated training data without header.
            Column order must match the order specified in self.feature_names.
          - test.csv: comma separated training data without header.
            Column order must match the order specified in self.feature_names.
          - mean_std.json: json dictionary of format {feature_name: [mean, std]},
            containing mean and std for numerical features. For example,
            "priors_count": [4.745, 1.34],...}.
          - vocabulary.json: json dictionary containing vocabulary for categorical
            features of format {feature_name: [feature_vocabulary]}. For example,
            {sex": ["Female", "Male"],...}.

        """
        
        self._dataset_base_dir = dataset_base_dir
        if train_file:
            self._train_file = train_file
        else:
            self._train_file = ["{}train_modified.csv".format(self._dataset_base_dir)]
        
        if test_file:
            self._test_file = test_file
        else:
            self._test_file = ["{}test_modified.csv".format(self._dataset_base_dir)]
        
        self._mean_std_file = "{}mean_std.json".format(self._dataset_base_dir)
        # self._vocabulary_file = "{}/vocabulary.json".format(self._dataset_base_dir)
        
        self.feature_names = [
            "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count",
            "age", "c_charge_degree", "c_charge_desc", "age_cat", "sex", "race",
            "is_recid"]
        
        # self.RECORD_DEFAULTS = [  # pylint: disable=invalid-name
        # [0.0], [0.0], [0.0], [0.0], [0.0], ["?"],
        # ["?"], ["?"], ["?"], ["?"], ["?"]]
        # self.RECORD_DEFAULTS = [  # pylint: disable=invalid-name
        #     [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
        #     [0.0], [0.0], [0.0], [0.0], [0.0]]
        
        self.RECORD_DEFAULTS = [  # pylint: disable=invalid-name
            [0.0], [0.0], [0.0], [0.0], [0.0], ["?"],
            ["?"], ["?"], [0.0], [0.0], [0.0]]
        
        self.target_column_name = "is_recid"
        # self.target_column_positive_value = "Yes"
        self.target_column_positive_value = 1
        
        # # Following params are tied to subgroups in targets["subgroup"]
        self.sensitive_column_names = ["sex", "race"]
        # self.sensitive_column_values = ["Female", "Black"]
        self.sensitive_column_values = [1, 1]
        
        # in this code female = 1, black = 1
        # in original white = 0 black = 1 male = 0 female = 1  the same
    
    def get_input_fn(self, mode):
        
        if mode == 'train':
            filename_queue = self._train_file
        
        else:
            filename_queue = self._test_file
        
        # Extracts basic features and targets from filename_queue
        
        features, targets, all_features = self.extract_features_and_targets(filename_queue)
        
        # dataloader = torch.utils.data.DataLoader(dataset=MyDataset(features,targets), batch_size=batch_size, num_workers=0)
        # Adds subgroup information to targets. Used to plot metrics.
        targets = self.add_subgroups_to_targets(targets, all_features)
        
        return features, targets
    
    def data_standarization(self, data):
        
        # print('data?',data.shape)
        
        data_std = np.array([list(map(int, data[i, :])) for i in range(data.shape[0])]).reshape(data.shape).astype(
            np.float)
        # print('data std',data_std)
        
        mean = np.mean(data_std, axis=0)
        # print('mean',mean)
        std = np.std(data_std, axis=0)
        # print('std',std)
        
        standard_data = (data_std - mean) / std
        # print('standard_data',standard_data.shape)
        return standard_data
    
    def extract_features_and_targets(self, filename_queue):
        """Extracts features and targets from filename_queue."""
        
        # extract features and targets from the train and test csv
        # now the features and targets are not values but texts
        # file = open(filename_queue,"r")
        # _, value = file.readline()  #key,value
        # feature_list = torch.tensor(value)  # 用的到吗？？？？？
        #
        # # Setting features dictionary.
        # features = dict(zip(self.feature_names, feature_list))
        # features = self._binarize_protected_features(features)   #需要吗
        # features = tf.train.batch(features, batch_size)  #转成batch
        
        # # Setting targets dictionary.
        # targets = {}
        # targets[self.target_column_name] = tf.reshape(
        #     tf.cast(
        #         tf.equal(
        #             features.pop(self.target_column_name),
        #             self.target_column_positive_value), tf.float32), [-1, 1])
        
        csv_file = open(filename_queue[0])  # 打开csv文件
        csv_reader_lines = csv.reader(csv_file)
        # print('csv_reader_lines',csv_reader_lines)
        data = []
        for lines in csv_reader_lines:
            # print('lines',lines)
            data = np.append(data, lines)
        
        
        data = data.reshape((-1, 11))

        categorial_features = data[:, 5:8]  # 5 6 7
        # print('dcategorial_features', categorial_features)

        str_1 = data[:, 5]
        # print('str_1',type(str_1))
        str_1 = np.array(list(map(int, str_1)))
        # print('str_1',str_1)
        str_2 = data[:, 6]
        str_2 = np.array(list(map(int, str_2)))
        str_3 = data[:, 7]
        str_3 = np.array(list(map(int, str_3)))

        embedding_1 = nn.Embedding(max(str_1) + 1, 32)
        str_to_categorial_1 = embedding_1(torch.tensor(str_1)).detach().numpy()
        # print('str_to_categorial_1 ',str_to_categorial_1)
        # print('str_to_categorial_1 ', str_to_categorial_1.shape)   #(5049, 32)
 

        embedding_2 = nn.Embedding(max(str_2) + 1, 32)
        str_to_categorial_2 = embedding_2(torch.tensor(str_2)).detach().numpy()
        # print('str_to_categorial_2 ', str_to_categorial_2.shape)

        embedding_3 = nn.Embedding(max(str_3) + 1, 32)
        str_to_categorial_3 = embedding_3(torch.tensor(str_3)).detach().numpy()
        # print('str_to_categorial_3 ', str_to_categorial_3.shape)

        input_feature = np.concatenate((data[:, 0:5], str_to_categorial_1, str_to_categorial_2, str_to_categorial_3 ,data[:, 8:11]),1)
        # print('1input_feature', input_feature.shape)


        # print('1data.shape',data.shape)
        input_feature = input_feature[input_feature[:, 102] != '2']
        # print('2input_feature', input_feature.shape)
        # print('2data.shape',data.shape)
        # feature without sensitive information
        all_features = input_feature[:, 0:103]
        print('1HERE')
        print('all_features',all_features)
        
        # print('all_features.shape', all_features.shape)
        features = np.delete(all_features, -1, axis=1)
        features = np.delete(features, -1, axis=1)
        
        targets =input_feature[:, -1]
        # print('3input_feature', input_feature.shape)    # 4298/1852
        # print('features.shape', features.shape)
        #
        features_std = self.data_standarization(features[:,0:5])
        features_final =  np.concatenate((features_std, features[:,5:101]),1)
        # features = self.data_standarization(features)

        # print('4features',features_final.shape)
        #
        print('2HERE')
        print('features_final',features_final)
        print('targets',targets)
 
        
        return features_final, targets, all_features
    
    def add_subgroups_to_targets(self, targets, all_features):
        """Adds subgroup information to targets dictionary."""
        
        # print('all_features',all_features.shape)
        # print('targets.shape',targets.shape)                                # GT label + (race + sex)
        targets = targets.reshape((-1, 1))
        sensitive_features = all_features[:, 101]
        targets = np.insert(targets, -1, values=sensitive_features, axis=1)
        
        sensitive_features = all_features[:, 102]
        targets = np.insert(targets, -1, values=sensitive_features, axis=1)
        
        return targets
 
            
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.data = features
        self.label = targets
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    def __len__(self):
        return len(self.data)


compasInput = CompasInput(dataset_base_dir, train_file=None, test_file=None)
print('#############train#################')
train_features, train_targets = compasInput.get_input_fn(mode='train')
# print('train_features',train_features.shape)
# print('train_targets',train_targets.shape)
print('#############test#################')
test_features, test_targets = compasInput.get_input_fn(mode='test')
# print('test_targets',test_targets)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = train_features
        self.label = train_targets[:, -1]
        self.sens = train_targets[:, 0:2]
        
        # print('self.label',self.label)
        # print('self.sens',self.sens)
    
    def __getitem__(self, idx):
        self.data_idx = np.array(list(map(float, self.data[idx])))
        self.label_idx = np.array(list(map(float, self.label[idx])))
        self.sens_idx = np.array(list(map(float, self.sens[idx])))
        
        return self.data_idx, self.label_idx, self.sens_idx
    
    def __len__(self):
        return len(self.data)


class MyDatasetTest(torch.utils.data.Dataset):
    def __init__(self):
        self.data = test_features
        self.label = test_targets[:, -1]
        self.sens = test_targets[:, 0:2]
    
    def __getitem__(self, idx):
        self.data_idx = np.array(list(map(float, self.data[idx])))
        self.label_idx = np.array(list(map(float, self.label[idx])))
        self.sens_idx = np.array(list(map(float, self.sens[idx])))
        
        return self.data_idx, self.label_idx, self.sens_idx
    
    def __len__(self):
        return len(self.data)


def data_set():
    
    dataset = MyDataset()
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=0,
                                             shuffle=True)
    dataset_test = MyDatasetTest()
    dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, num_workers=0,
                                                  shuffle=False)
    
    # len_train_data = train_features.shape[0]
    # weights_train = [1] * len_train_data
    # sampler_train = WeightedRandomSampler(weights_train, len_train_data, replacement=True)
    # dataset = MyDataset()
    # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, num_workers=0,
    #                                          sampler=sampler_train)
    # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=0,
    #                                          shuffle=True)
    
    # len_test_data = test_features.shape[0]
    # weights_test = [1] * len_test_data
    # sampler_test = WeightedRandomSampler(weights_test, len_test_data, replacement=True)
    # dataset_test = MyDatasetTest()
    # dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=1852, num_workers=0,
    #                                               sampler=sampler_test)
    # dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, num_workers=0,
    #                                               shuffle=False)
 
    return dataloader, dataloader_test

# 5049 2165
# 4298/1852

# print(torch.__version__)
