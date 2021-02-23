# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:13:08 2021

@author: Manuel Camargo
"""
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import itertools
from operator import itemgetter

import utils.support as sup
import readers.log_reader as lr
import readers.log_splitter as ls
from nltk.util import ngrams


class Input:
    #Class variables (remember that they are different than instance variables, and all instances or objects have access to them)
    path = '' #The location where the results will be written
    mode = ''   #Type of prediction task that the object will be used for, i.e., "event_prediction", "timestamp_prediction", "event_timestamp_prediction"
    dataset_name = '' #Name of the input dataset
    prefix_len = ''  #It is a number that shows the length of the considered prefixes
    batch = ''       #It is a number that shows size of used batch
    design_matrix = ''  # A matrix that stores the designed matrix (each activity is shown by one hot vector)
    design_matrix_padded = '' #A design matrix that is padded after creating the prefixes
    y = '' #The ground truth labels related to the "design_matrix_padded"
    unique_event = ''  #The list of unique events, including end of trace as "0"
    selected_columns = '' # List of considered columns, including event and other information
    timestamp_loc = ''    # The column index for timestamp feature
    train_inds=''     #Index of training instances
    test_inds=''      #Index of test instances
    validation_inds=''     #Index of validation instances
    train_loader = ''
    test_loader = ''
    validation_loader = ''




    #class methods can be called without creating objects (they have cls instead of self)
    #start from here
    @classmethod
    def run(cls, param):
        '''
        This method is the starting point for preparing an object to be used later in different prediction tasks.

        @param path: The location of the event log
        @param prefix: Size of the prefix
        @param batch_size: Size of batch
        @param mode: "event_prediction", "timestamp_prediction", "event_timestamp_prediction"
        @return:
        '''
        cls.param = param
        cls.prefix_len = param['prefix']
        cls.batch = param['batch_size']
        cls.mode = param['mode']
        cls.path = param['output_path']
        cls.file_name = param['file_name'].split('.')[0]
        # #Reading a file
        log = cls.__load_log(param)
        # Split log in train/test
        log_train, cls.log_test = cls.__split_timeline(log, 0.8, param['one_timestamp'])
        # indexing
        log_train, cls.ac_index = cls.__indexing(log_train)
        cls.unique_event = list({v: k for k, v in cls.ac_index.items()}.keys())
        print("Original data:", log_train.head())
        # Split log in train/validation
        log_train, log_valdn = cls.__split_timeline(log_train, 0.8, param['one_timestamp'])
        # Process data
        train_data, y_train = cls.__create_design_matix(log_train)
        # print(train_data.size())
        cls.train_loader = DataLoader(
            dataset=TensorDataset(train_data, y_train), 
            batch_size=cls.batch, 
            shuffle=True)

        valdn_data, y_valdn = cls.__create_design_matix(log_valdn)
        cls.validation_loader = DataLoader(
            dataset=TensorDataset(valdn_data, y_valdn), 
            batch_size=cls.batch, 
            shuffle=True)
        
        cls.__export_parms()
 

    @classmethod
    def __create_design_matix(cls, partition):
        if cls.mode == 'event_two_timestamps':
            data_augment = cls.__create_data_augmented(partition, cls.param)
        else:
            data_augment = cls.__create_data_augmented_original(partition, cls.param)
        # Creating a design matrix that shows one hot vector representation for activity IDs
        # design_matrix = cls.__design_matrix_creation(data_augment, cls.mode)
        # # Creating prefix
        # design_matrix_padded, y =  cls.__prefix_creating(design_matrix)
        design_matrix_padded, y =  cls._vectorize_seq(data_augment)
        print("The dimension of designed matrix:", design_matrix_padded.size())
        print("The dim of ground truth:", y.size())
        print("The prefix considered so far:", design_matrix_padded.size()[1])
        return design_matrix_padded, y
    
    
    @staticmethod
    def __create_data_augmented_original(data, params):
        # Calculate times
        data_augment = list()
        for name, gr in tqdm(data.groupby('caseid'), 'calculating times:'):
            # sorting by time
            if params['tm_type'] == 'start':
                gr.sort_values(by=['start_timestamp'])
                duration_time = gr.loc[:, 'start_timestamp'].diff() / np.timedelta64(1, 'D')
            else:
                gr.sort_values(by=['end_timestamp'])
                duration_time = gr.loc[:, 'end_timestamp'].diff() / np.timedelta64(1, 'D')
            # Filling Nan with 0
            duration_time.iloc[0] = 0
            # computing the remaining time
            length = duration_time.shape[0]
            remaining_time = [np.sum(duration_time[i + 1:length]) for i in range(duration_time.shape[0])]
            gr['duration_time'] = duration_time
            gr['remaining_time'] = remaining_time
            data_augment.append(gr)
        data_augment = pd.concat(data_augment, axis=0)
        return data_augment

    @staticmethod
    def __create_data_augmented(data, params):
        """Appends the indexes and relative time to the dataframe.
        parms:
            log: dataframe.
        Returns:
            Dataframe: The dataframe with the calculated features added.
        """
        data['duration_time'] = 0
        data['waiting_time'] = 0
        data['remaining_time'] = 0
        data = data.to_dict('records')
        data = sorted(data, key=lambda x: x['caseid'])
        for _, group in itertools.groupby(data, key=lambda x: x['caseid']):
            events = list(group)
            last_event_ts = events[-1]['end_timestamp']
            ordk = 'start_timestamp'
            events = sorted(events, key=itemgetter(ordk))
            for i in range(0, len(events)):
                # In one-timestamp approach the first activity of the trace
                # is taken as instantsince there is no previous timestamp
                # to find a range
                dur = (events[i]['end_timestamp'] -
                        events[i]['start_timestamp']) / np.timedelta64(1, 'D')
                rt = (last_event_ts - events[i]['start_timestamp']) / np.timedelta64(1, 'D')
                if i == 0:
                    wit = 0
                else:
                    wit = (events[i]['start_timestamp'] -
                            events[i-1]['end_timestamp']) / np.timedelta64(1, 'D')
                events[i]['waiting_time'] = wit if wit >= 0 else 0
                events[i]['duration_time'] = dur
                events[i]['remaining_time'] = rt

        return pd.DataFrame.from_dict(data)
    
    @classmethod
    def __indexing(cls, log):
        
        def create_index(log_df, column):
            """Creates an idx for a categorical attribute.
            parms:
                log_df: dataframe.
                column: column name.
            Returns:
                index of a categorical attribute pairs.
            """
            temp_list = log_df[[column]].values.tolist()
            subsec_set = {(x[0]) for x in temp_list}
            subsec_set = sorted(list(subsec_set))
            alias = dict()
            for i, _ in enumerate(subsec_set):
                alias[subsec_set[i]] = i + 1
            return alias
    
        # Activities index creation
        ac_index = create_index(log, 'task')
        ac_index['start'] = 0
        ac_index['end'] = len(ac_index)
        # Add index to the event log
        ac_idx = lambda x: ac_index[x['task']]
        log['ac_index'] = log.apply(ac_idx, axis=1)
        return log, ac_index
    
    @classmethod
    def __load_log(cls, params):
        params['read_options']['filter_d_attrib'] = False
        log = lr.LogReader(os.path.join(params['input_path'], params['file_name']),
                           params['read_options'])
        log_df = pd.DataFrame(log.data)
        if set(['Unnamed: 0', 'role']).issubset(set(log_df.columns)):
            log_df.drop(columns=['Unnamed: 0', 'role'], inplace=True)
        log_df = log_df[~log_df.task.isin(['Start', 'End'])]
        return log_df
    
    @classmethod
    def __design_matrix_creation(cls, data_augment, mode):
        '''
        data_augment is pandas dataframe created after reading CSV input by "read_csv()" method
        '''
        # Creating a desing matrix (one hot vectors for activities), End of line (case) is denoted by class 0
        unique_event = sorted(list(cls.ac_index.values()))
        # Manuel Camargo: Adition of start and end event
        # unique_event = unique_event + [len(unique_event)+1]
        # unique_event = unique_event + [len(unique_event)]
        # print("uniqe events:", unique_event)
    
        l = []
        for index, row in tqdm(data_augment.iterrows(), 
                               'creating design marix:'):
            temp = dict()
            '''
            temp ={1: 0,
                  2: 0,
                  3: 1,
                  4: 0,
                  5: 0,
                  6: 0,
                  '0':0,
                  'duration_time': 0.0,
                  'remaining_time': 1032744.0}
            '''
    
            # Defning the columns we consider
            if mode == 'event_two_timestamps':
                cols = ['duration_time', 'waiting_time', 'remaining_time']
            else:
                cols = ['duration_time', 'remaining_time']
            keys = list(unique_event) + cols
            for k in keys:
                if (k == row['ac_index']):
                    temp[k] = 1
                else:
                    temp[k] = 0
            temp['class'] = row['ac_index']
            temp['duration_time'] = row['duration_time']
            if mode == 'event_two_timestamps':
                temp['waiting_time'] = row['waiting_time']
            temp['remaining_time'] = row['remaining_time']
            temp['CaseID'] = row['caseid']
    
            l.append(temp)
    
        # Creating a dataframe for dictionary l
        design_matrix = pd.DataFrame(l)
        print("The design matrix is:\n", design_matrix.head(10))
        return design_matrix
    
    # Creating the desing matrix based on given prefix.
    @classmethod
    def __prefix_creating(cls, data_matrix):


        if (cls.mode == "timestamp_prediction"):
            clsN = data_matrix.columns.get_loc('duration_time')
        elif (cls.mode == "event_prediction"):
            clsN = data_matrix.columns.get_loc('class')
        elif (cls.mode == 'event_timestamp_prediction'):
            clsN = [data_matrix.columns.get_loc('duration_time')] + [data_matrix.columns.get_loc('class')]
            cls.timestamp_loc = data_matrix.columns.get_loc('duration_time')
            cls.selected_columns = cls.unique_event + [cls.timestamp_loc]
        elif (cls.mode == 'event_two_timestamps'):
            clsN = [data_matrix.columns.get_loc('duration_time'), 
                    data_matrix.columns.get_loc('waiting_time'),
                    data_matrix.columns.get_loc('class')]
            cls.timestamp_loc = data_matrix.columns.get_loc('duration_time')
            cls.waiting_time_loc = data_matrix.columns.get_loc('waiting_time')
            cls.selected_columns = cls.unique_event + [cls.timestamp_loc, 
                                                        cls.waiting_time_loc]
        # data_matrix = data_matrix[data_matrix.CaseID=='1']  
        group = data_matrix.groupby('CaseID')
        # Iterating over the groups to create tensors
        temp = []
        temp_shifted = []
        for name, gr in group:
            gr = gr.drop('CaseID', axis=1)
            # For each group, i.e., view, we create a new dataframe and reset the index
            gr = gr.copy(deep=True)
            gr = gr.reset_index(drop=True)
            
            # adding a new row at the start and bottom of each case to denote the end of a case
            new_row = pd.DataFrame([{k: 0 for k in gr.columns}])
            for _ in range(0, cls.prefix_len):
                gr = pd.concat([new_row, gr], axis=0, ignore_index=True)
            gr = pd.concat([gr, new_row], axis=0, ignore_index=True)
            
            # Modification Manuel Camargo: Start of line is denoted by class 0 and is left padded at start
            remaining_time = gr.iloc[cls.prefix_len, gr.columns.get_loc('remaining_time')]
            for i in range(0, cls.prefix_len):
                gr.iloc[i, gr.columns.get_loc(cls.ac_index['start'])] = 1  # End of line is denoted by class 0
                gr.iloc[i, gr.columns.get_loc('remaining_time')] = remaining_time  # End of line is denoted by class 0
            # Modification Manuel Camargo: End of line is denoted by class num_clases+1
            gr.iloc[gr.shape[0] - 1, gr.columns.get_loc(cls.ac_index['end'])] = 1  # End of line is denoted by class 0
            gr.iloc[gr.shape[0] - 1, gr.columns.get_loc('class')] = cls.ac_index['end']  # End of line is denoted by class 0
            # Selecting only traces that has length greater than the defined prefix
            for i in range(gr.shape[0]):
                # if (i+prefix == gr.shape[0]):
                #   break
                temp.append(
                    torch.tensor(gr.iloc[i:i + cls.prefix_len].values, 
                                  dtype=torch.float, 
                                  requires_grad=False))
    
                # Storing the next element after the prefix as the prediction class
                try:
                    # print("the prediction:", "the i", i ,gr.iloc[i+prefix,cls])
                    temp_shifted.append(
                        torch.tensor([gr.iloc[i + cls.prefix_len, clsN]], 
                                      dtype=torch.float, 
                                      requires_grad=False))
                except IndexError:
                    # Printing the end of sequence
                    # print("the prediction:", "ESLE the i", i ,0)
                    temp_shifted.append(
                        torch.tensor([np.float16(cls.ac_index['end'])], 
                                      dtype=torch.float, 
                                      requires_grad=False))
        design_matrix_padded = pad_sequence(temp, batch_first=True)
        design_matrix_shifted_padded = pad_sequence(temp_shifted, batch_first=True)
        #Applying pad corrections
        for i in range(design_matrix_padded.size()[0]):
            u = (design_matrix_padded[i, :, cls.ac_index['end']] == 1).nonzero()
            try:
                design_matrix_padded[i, :, cls.ac_index['end']][u:] = 1
            except TypeError:
                pass
        return design_matrix_padded, design_matrix_shifted_padded

    @classmethod
    def _vectorize_seq(cls, data_matrix):
        
        """
        Dataframe vectorizer.
        parms:
            columns: list of features to vectorize.
            parms (dict): parms for training the network
        Returns:
            dict: Dictionary that contains all the LSTM inputs.
        """
        def to_categorical(y, num_classes=None, dtype='float32'):
            y = np.array(y, dtype='int')
            input_shape = y.shape
            if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
                input_shape = tuple(input_shape[:-1])
            y = y.ravel()
            if not num_classes:
                num_classes = np.max(y) + 1
            n = y.shape[0]
            categorical = np.zeros((n, num_classes), dtype=dtype)
            categorical[np.arange(n), y] = 1
            output_shape = input_shape + (num_classes,)
            categorical = np.reshape(categorical, output_shape)
            return categorical
        
        cls.timestamp_loc = len(cls.unique_event)
        cls.waiting_time_loc = len(cls.unique_event)+1
        cls.selected_columns = cls.unique_event + [cls.timestamp_loc, 
                                                    cls.waiting_time_loc]
        columns = ['ac_index', 'duration_time', 'waiting_time']
        times = ['duration_time', 'waiting_time']
        x_ac_list = list()
        y_ac_list = list()
        x_times_dict = dict()
        y_times_dict = dict()
        # reformat_events(log, columns, ac_index)
        data_matrix = cls.reformat_events(data_matrix, columns, cls.ac_index)
        # n-gram definition
        for i, _ in enumerate(data_matrix):
            for x in columns:
                serie = list(ngrams(data_matrix[i][x], cls.prefix_len,
                                    pad_left=True, left_pad_symbol=0))
                y_serie = [x[-1] for x in serie]
                serie = serie[:-1]
                y_serie = y_serie[1:]
                if x == 'ac_index':
                    x_ac_list = (
                        x_ac_list + serie if i > 0 else serie)
                    y_ac_list = (
                        y_ac_list + y_serie if i > 0 else y_serie)
                elif x in times:
                    x_times_dict[x] = (
                        x_times_dict[x] + serie if i > 0 else serie)
                    y_times_dict[x] = (
                        y_times_dict[x] + y_serie if i > 0 else y_serie)
        # Transform task, dur and role prefixes in vectors
        x_ac_list  = np.array(x_ac_list)
        y_ac_array  = np.array(y_ac_list)
        x_ac_array = to_categorical(x_ac_list, num_classes=len(cls.ac_index))
        y_ac_array = y_ac_array.reshape(y_ac_array.shape[0], 1)
        # reshape times
        for key, value in x_times_dict.items():
            x_times_dict[key] = np.array(value)
            x_times_dict[key] = x_times_dict[key].reshape(
                (x_times_dict[key].shape[0], x_times_dict[key].shape[1], 1))
        x_times_array = np.dstack(list(x_times_dict.values()))
        y_times_array = np.dstack(list(y_times_dict.values()))[0]
        x_array = np.concatenate((x_ac_array, x_times_array), axis=2)
        y_array = np.concatenate((y_times_array, y_ac_array), axis=1)
        y_array = y_array.reshape((y_array.shape[0], 1 , y_array.shape[1]))
        x_array = torch.tensor(x_array, dtype=torch.float, requires_grad=False)
        y_array= torch.tensor(y_array, dtype=torch.float, requires_grad=False)
        return x_array, y_array

   
    @staticmethod
    def __split_timeline(data: pd.DataFrame, size: float, one_ts: bool) -> None:
        """
        Split an event log dataframe by time to peform split-validation.
        prefered method time splitting removing incomplete traces.
        If the testing set is smaller than the 10% of the log size
        the second method is sort by traces start and split taking the whole
        traces no matter if they are contained in the timeframe or not

        Parameters
        ----------
        size : float, validation percentage.
        one_ts : bool, Support only one timestamp.
        """
        # Split log data
        splitter = ls.LogSplitter(data)
        train, test = splitter.split_log('timeline_contained', size, one_ts)
        total_events = len(data)
        # Check size and change time splitting method if necesary
        if len(test) < int(total_events*0.1):
            train, test = splitter.split_log('timeline_trace', size, one_ts)
        # Set splits
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        test = pd.DataFrame(test)
        train = pd.DataFrame(train)
        log_test = (test.sort_values(key, ascending=True)
                         .reset_index(drop=True))
        log_train = (train.sort_values(key, ascending=True)
                         .reset_index(drop=True))
        return log_train, log_test

    @classmethod
    def __export_parms(cls):
        if not os.path.exists(os.path.join(cls.param['output_path'], 'parameters')):
            os.makedirs(os.path.join(cls.param['output_path'], 'parameters'))

        cls.param['max_trace_size'] = int(cls.log_test.groupby('caseid')['task']
                                      .count().max())
        
        cls.param['ac_index'] = cls.ac_index
        
        sup.create_json(cls.param, os.path.join(cls.param['output_path'],
                                            'parameters',
                                            'model_parameters.json'))

    # =============================================================================
    # Reformat events
    # =============================================================================
    @classmethod
    def reformat_events(cls, log, columns, ac_index):
        """Creates series of activities, roles and relative times per trace.
        parms:
            self.log: dataframe.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
        Returns:
            list: lists of activities, roles and relative times.
        """
        temp_data = list()
        log_df = log.to_dict('records')
        key =  'start_timestamp'
        log_df = sorted(log_df, key=lambda x: (x['caseid'], key))
        for key, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
            trace = list(group)
            temp_dict = dict()
            for x in columns:
                serie = [y[x] for y in trace]
                if x == 'ac_index':
                    serie.insert(0, ac_index[('start')])
                    serie.append(ac_index[('end')])
                else:
                    serie.insert(0, 0)
                    serie.append(0)
                temp_dict = {**{x: serie}, **temp_dict}
            temp_dict = {**{'caseid': key}, **temp_dict}
            temp_data.append(temp_dict)
        return temp_data