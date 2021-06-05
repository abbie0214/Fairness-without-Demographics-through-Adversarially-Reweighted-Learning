from __future__ import division
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import json
import os,sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
import torch
import csv
from torch.utils.data import DataLoader

import json

pd.options.display.float_format = '{:,.2f' \
                                  '}'.format
dataset_base_dir = './data/uci_adult/'

def convert_object_type_to_category(df):
  """Converts columns of type object to category."""
  df = pd.concat([df.select_dtypes(include=[], exclude=['object']),
                  df.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
                  ], axis=1).reindex(df.columns, axis=1)
  return df

TRAIN_FILE = os.path.join(dataset_base_dir,'adult.data')
TEST_FILE = os.path.join(dataset_base_dir,'adult.test')

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

target_variable = "income"
target_value = ">50K"

with open(TRAIN_FILE, "r") as TRAIN_FILE:
  train_df = pd.read_csv(TRAIN_FILE,sep=',',names=columns)

with open(TEST_FILE, "r") as TEST_FILE:
  test_df = pd.read_csv(TEST_FILE,sep=',',names=columns)

# Convert columns of type ``object`` to ``category``
train_df = convert_object_type_to_category(train_df)
test_df = convert_object_type_to_category(test_df)
print('train_df',train_df)
print('test_df',test_df)


# print('???', len(np.where(train_df['workclass'] == ' ?')[0]))
# print('???', len(np.where(test_df['workclass'] == ' ?')[0]))

# Binarize target_variable and sensitive label sex
train_df['income'] = train_df.apply(lambda x: 1 if x['income']==' >50K' else 0, axis=1).astype('category')
test_df['income'] = test_df.apply(lambda x: 1 if x['income']==' >50K.' else 0, axis=1).astype('category')

train_df['sex'] = train_df.apply(lambda x: 1 if x['sex']==' Female' else 0, axis=1).astype('category')
test_df['sex'] = test_df.apply(lambda x: 1 if x['sex']==' Female' else 0, axis=1).astype('category')


output_file_path = os.path.join(dataset_base_dir,'train.csv')
with open(output_file_path, mode="w",newline='') as output_file:
    train_df.to_csv(output_file,index=False,columns=columns,header=False)
    output_file.close()

output_file_path = os.path.join(dataset_base_dir,'test.csv')
with open(output_file_path, mode="w",newline='') as output_file:
    test_df.to_csv(output_file,index=False,columns=columns,header=False)
    output_file.close()

    
    #######################################################################################################
#
# train: test = 32560:16281
#
# Construct vocabulary.json, and write to directory.¶
# vocabulary.json: json dictionary of the format {feature_name: [feature_vocabulary]}, containing vocabulary for categorical features.
#
# cat_cols = train_df.select_dtypes(include='category').columns
# vocab_dict = {}
# for col in cat_cols:
#     vocab_dict[col] = list(set(train_df[col].cat.categories) - {"?"})
#
# output_file_path = os.path.join(dataset_base_dir, 'vocabulary.json')
# with open(output_file_path, mode="w") as output_file:
#     output_file.write(json.dumps(vocab_dict))
#     output_file.close()
# # print(vocab_dict)
#
#
# temp_dict = train_df.describe().to_dict()
# mean_std_dict = {}
# for key, value in temp_dict.items():
#   mean_std_dict[key] = [value['mean'],value['std']]
#
# output_file_path = os.path.join(dataset_base_dir,'mean_std.json')
# with open(output_file_path, mode="w") as output_file:
#     output_file.write(json.dumps(mean_std_dict))
#     output_file.close()
# print(mean_std_dict)
#
#
#
# SUBGROUP_TARGET_COLUMN_NAME = "subgroup"


class UCIAdultInput():
    """Data reader for UCI Adult dataset."""

    def __init__(self,
               dataset_base_dir,
               train_file=None,
               test_file=None):
        """Data reader for UCI Adult dataset.
        Args:
          dataset_base_dir: (string) directory path.
          train_file: string list of training data paths.
          test_file: string list of evaluation data paths.
          dataset_base_sir must contain the following files in the dir:
          - train.csv: comma separated training data without header.
            Column order must match the order specified in self.feature_names.
          - test.csv: comma separated training data without header.
            Column order must match the order specified in self.feature_names.


        """
    # pylint: disable=long-line,line-too-long

        self._dataset_base_dir = dataset_base_dir
        if train_file:
            self._train_file = train_file
        else:
            self._train_file = ["{}train_modified.csv".format(self._dataset_base_dir)]


        if test_file:
            self._test_file = test_file
        else:
            self._test_file = ["{}test_modified.csv".format(self._dataset_base_dir)]


        # self._mean_std_file = "{}/mean_std.json".format(self._dataset_base_dir)
        # self._vocabulary_file = "{}/vocabulary.json".format(self._dataset_base_dir)
        #

        self.feature_names = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country",
            "income"]

        # self.RECORD_DEFAULTS = [[0.0], ["?"], [0.0], ["?"], [0.0], ["?"], ["?"],  # pylint: disable=invalid-name
        #                         ["?"], ["?"], ["?"], [0.0], [0.0], [0.0], ["?"],
        #                         ["?"]]

        # Initializing variable names specific to UCI Adult dataset input_fn
        self.target_column_name = "income"
        self.target_column_positive_value = 1
        self.sensitive_column_names = ["sex", "race"]
        self.sensitive_column_values = [1, 1]

        # self.weight_column_name = "instance_weight"

    def get_input_fn(self, mode):
        """Gets input_fn for UCI census income data.
        Args:
          mode: The execution mode, as defined in tf.estimator.ModeKeys.
          batch_size: An integer specifying batch_size.
        Returns:
          An input_fn.
        """

        if mode == 'train':
            filename_queue = self._train_file

        else:
            filename_queue = self._test_file

        # Extracts basic features and targets from filename_queue
        features, targets, all_features = self.extract_features_and_targets(filename_queue)

        # Adds subgroup information to targets. Used to plot metrics.
        targets = self.add_subgroups_to_targets(targets, all_features)

        
        return features, targets


    def data_standarization(self, data):
        # print('data?',data.shape)

        data_std = np.array([list(map(int, data[i, :])) for i in range(data.shape[0])]).reshape(data.shape).astype(np.float)
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


        data = data.reshape((-1, 15))
        
        print('data',data)
        
        # trr = data[data[:,13] != '2']
        # tr = trr[trr[:,1] != '5']
        # print('tr.shape',tr.shape)
        #
        #
        # print('data.shape',data.shape)
        # print('1data.shape',data.shape)
        data =data[data[:,13] != '2']
        
        # print('2data.shape',data.shape)
        
 
        # feature without sensitive information
        all_features = data[:,0:14]
        features = np.delete(all_features, -1, axis=1)
        features = np.delete(features, -1, axis=1)

        targets = data[:,-1]
        # print('data', data.shape)    # 4298/1852
        # print('features.shape', features.shape)

        features = self.data_standarization(features)
        #
        # print('features',features)


        return features, targets, all_features




    def add_subgroups_to_targets(self, targets, all_features):
        """Adds subgroup information to targets dictionary."""

        # print('all_features',all_features.shape)
        # print('targets.shape',targets.shape)                                # GT label + (race + sex)
        targets = targets.reshape((-1,1))
        sensitive_features = all_features[:,12]
        targets = np.insert(targets,-1,values = sensitive_features,axis = 1)

        sensitive_features = all_features[:,13]
        targets = np.insert(targets,-1,values = sensitive_features,axis = 1)


        return targets


class MyDataset(torch.utils.data.Dataset):
    def __init__(self,features,targets):
        self.data = features
        self.label = targets


    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)



uciInput = UCIAdultInput(dataset_base_dir,train_file=None,test_file=None)
train_features, train_targets = uciInput.get_input_fn(mode = 'train')
test_features, test_targets = uciInput.get_input_fn(mode = 'test')


# print('train_features',train_features)
# print(' train_targets ', train_targets )
# print('test_features',test_features)
# print(' test_targets ', test_targets )

# print('############################################')
# print('train_features',train_features)   #31522, 12
# print('train_features.shape',train_features.shape)
print('train_targets',train_targets)#31522, 3
# print('train_targets.shape',train_targets.shape)
#
# print('############################################')
# print('test_features',test_features)   #15801, 12
# print('test_features.shape',test_features.shape)
print('test_targets',test_targets)#15801, 3
# print('test_targets.shape',test_targets.shape)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = train_features
        self.label = train_targets[:,-1]
        self.sens = train_targets[:,0:2]

        # print('self.label',self.label)
        # print('self.sens',self.sens)
    def __getitem__(self, idx):

        self.data_idx = np.array(list(map(float, self.data[idx])))
        self.label_idx  = np.array(list(map(float, self.label[idx])))
        self.sens_idx = np.array(list(map(float, self.sens[idx])))


        return self.data_idx,self.label_idx,self.sens_idx

    def __len__(self):
        return len(self.data)

class MyDatasetTest(torch.utils.data.Dataset):
    def __init__(self):
        self.data = test_features
        self.label = test_targets[:,-1]
        self.sens = test_targets[:,0:2]


    def __getitem__(self, idx):
        self.data_idx = np.array(list(map(float, self.data[idx])))
        self.label_idx  = np.array(list(map(float, self.label[idx])))
        self.sens_idx = np.array(list(map(float, self.sens[idx])))


        return self.data_idx,self.label_idx,self.sens_idx

    def __len__(self):
        return len(self.data)


def data_set():
    # len_train_data = train_features.shape[0]
    # weights_train = [1] * len_train_data
    # sampler_train = WeightedRandomSampler(weights_train, len_train_data, replacement=True)
    dataset = MyDataset()
    # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=512, num_workers=0,
    #                                          sampler=sampler_train)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=512, num_workers=0,
                                             shuffle = True)

    # len_test_data = test_features.shape[0]
    # weights_test = [1] * len_test_data
    # sampler_test = WeightedRandomSampler(weights_test, len_test_data, replacement=True)
    dataset_test = MyDatasetTest()
    # dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=512, num_workers=0,
    #                                               sampler=sampler_test)
    dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=5267, num_workers=0,
                                                  shuffle = False)

    # dataset = MyDataset()
    # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size= 32, num_workers=0)
    #
    # dataset_test = MyDatasetTest()
    # dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test,batch_size=32, num_workers=0)

    return dataloader, dataloader_test

# for ?s:
# ??? 1836
# ??? 963

# for 2s:
# data.shape (32561, 15)
# 2data.shape (31522, 15)
# data.shape (16281, 15)
# 2data.shape (15801, 15)

#for ? and 2s together:
# tr.shape (29751, 15)
# tr.shape (14869, 15)
