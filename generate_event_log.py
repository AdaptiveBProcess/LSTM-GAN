# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 10:15:34 2021

@author: Manuel Camargo
"""
import os
import sys
import getopt
import json
import math
from tqdm import tqdm
import time
import copy

import datetime
from datetime import timedelta

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import itertools
import traceback
import multiprocessing
from multiprocessing import Pool

import readers.log_reader as lr
import analyzers.sim_evaluator as ev
import utils.support as sup
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


class GanEventLogGenerator():
    """
    This is the man class encharged of the model evaluation
    """

    def __init__(self, parms):
        self.output_route = os.path.join('output_files', parms['folder'])
        # load parameters
        self.load_parameters()
        self.parms['variant'] = parms['variant']
        # load log test
        self.log = self.load_log_test()
        # Calculate number of traces
        self.num_cases = len(self.log.caseid.unique())
        # self.num_cases = 5
        
    def load_parameters(self):
        # Loading of parameters from training
        path = os.path.join(self.output_route,
                            'parameters',
                            'model_parameters.json')
        with open(path) as file:
            self.parms = json.load(file)
            self.parms['index_ac'] = {
                v: k for k, v in self.parms['ac_index'].items()}
            file.close()


    def load_log_test(self):
        df_test = lr.LogReader(
            os.path.join(self.output_route,
                         'tst_'+self.parms['file_name'].split('.')[0]+'.csv'),
            self.parms['read_options'])
        df_test = pd.DataFrame(df_test.data)
        df_test = df_test[~df_test.task.isin(['Start', 'End'])]
        return df_test


    def generate(self):
        def pbar_async(p, msg):
            pbar = tqdm(total=reps, desc=msg)
            processed = 0
            while not p.ready():
                cprocesed = (reps - p._number_left)
                if processed < cprocesed:
                    increment = cprocesed - processed
                    pbar.update(n=increment)
                    processed = cprocesed
            time.sleep(1)
            pbar.update(n=(reps - processed))
            p.wait()
            pbar.close()

        cpu_count = multiprocessing.cpu_count()
        batch_size = self.parms['batch_size']
        r_num_cases = math.ceil(self.num_cases / batch_size)
        num_digits = len(str(r_num_cases * batch_size))
        rounded_cases = ['Case'+str(i).zfill(num_digits) 
                          for i in range(0, r_num_cases * batch_size)]
        b_size = math.ceil(len(rounded_cases) / r_num_cases)
        chunks = [rounded_cases[x:x+b_size] 
                  for x in range(0, len(rounded_cases), b_size)]
        reps = len(chunks)
        pool = Pool(processes=cpu_count)
        # Generate
        args = [(self.output_route, self.parms, cases) for cases in chunks]
        # event_log = self.generate_batch_trace(args[0])
        p = pool.map_async(self.generate_batch_trace, args)
        pbar_async(p, 'generating traces:')
        pool.close()
        # Save results
        event_log = list(itertools.chain(*p.get()))
        event_log = pd.DataFrame(event_log).sort_values('caseid')
        # print(event_log)
        keep_cases = list(event_log.caseid.unique())[:self.num_cases]
        event_log = event_log[event_log.caseid.isin(keep_cases)]
        event_log = event_log[~event_log.task.isin(['start', 'end'])]
        return event_log


    @staticmethod
    def generate_batch_trace(args):

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
        
        def decode_traces(parms, traces):
            """Example function with types documented in the docstring.
            Args:
                trace (list): trace of predicted events.
                case (int): case number.
            Returns:
                list: predicted business trace decoded.
            """
            log_trace = list()
            data = sorted(traces, key=lambda x: x['caseid'])
            for caseid, group in itertools.groupby(data, key=lambda x: x['caseid']):
                for i, event in enumerate(group):
                    # event = trace[i]
                    if i == 0:
                        now = datetime.datetime.now()
                        now.strftime(parms['read_options']['timeformat'])
                        start_time = (now + timedelta(days=event['wait']))
                    else:
                        start_time = (log_trace[i-1]['end_timestamp'] +
                                      timedelta(days=event['wait']))
                    end_time = (start_time + timedelta(days=event['dur']))
                    log_trace.append(dict(caseid=caseid,
                                          task=parms['index_ac'][event['task']],
                                          # role=parms['index_rl'][event[1]],
                                          start_timestamp=start_time,
                                          end_timestamp=end_time))
    
            return log_trace
        
        def gen(output_route, parms, cases):
            try:
                model_path = os.path.join(output_route,
                                          parms['file_name'].split('.')[0]+'_rnnG.m')
                model = torch.load(model_path)
                model.eval()
                batch_size = parms['batch_size']
                prefix_len = parms['prefix']
                num_features = len(parms['ac_index'])+2
                events = list(parms['ac_index'].values())
                events_array = np.zeros((batch_size, 1), dtype=np.float64)
                durations_array = np.zeros((batch_size, 1), dtype=np.float64)
                waitings_array = np.zeros((batch_size, 1), dtype=np.float64)
                x_ngram = np.zeros((batch_size, prefix_len, num_features))
                x_ngram[:,:,0] = 1
                generated_event_log = list()
                for i in range(0, math.ceil(parms.get('max_trace_size') / prefix_len)):
                    y_pred = model(torch.tensor(x_ngram,
                                                dtype=torch.float,
                                                requires_grad=False))
                    y_pred_event = F.softmax(y_pred[:, :, events], dim=2)
                    if parms['variant'] == 'arg_max':
                        y_pred_event = torch.argmax(y_pred_event, dim=2).cpu().numpy().astype(int)
                    else:
                        y_pred_event = y_pred_event.data.cpu().numpy()
                        preds = list()
                        for y in y_pred_event:
                            batch = list()
                            for x in y:
                                batch.append(np.random.choice(np.arange(0, len(x)), p=x))
                            preds.append(batch)
                        y_pred_event = np.array(preds)
                    events_array = np.append(events_array, y_pred_event, axis=1)
                    # Converting the labels into one hot encoding
                    y_pred_last_one_hot = to_categorical(y_pred_event, 
                                                           len(events))
                    y_pred_timestamp = (
                        y_pred[:, :, len(events)].data.cpu().numpy())
                    y_pred_waiting_time = (
                        y_pred[:, :, len(events)+1].data.cpu().numpy())
                    y_pred_timestamp[y_pred_timestamp < 0] = 0
                    y_pred_waiting_time[y_pred_waiting_time < 0] = 0
                    durations_array = np.append(durations_array,
                                                y_pred_timestamp, axis=1)
                    waitings_array = np.append(waitings_array, 
                                                y_pred_waiting_time, axis=1)
                    org_shape = (y_pred_timestamp.shape[0], 
                                 y_pred_timestamp.shape[1], 1)
                    predict_value = torch.cat((
                        torch.tensor(y_pred_last_one_hot,
                                      dtype=torch.float, requires_grad=False),
                        torch.tensor(y_pred_timestamp.reshape(org_shape),
                                      dtype=torch.float, requires_grad=False),
                        torch.tensor(y_pred_waiting_time.reshape(org_shape),
                                      dtype=torch.float, requires_grad=False)), 
                        dim=2)
                    x_ngram = predict_value.data.cpu().numpy()
                for trace, durations, waitings, case in zip(events_array, 
                                                      durations_array, 
                                                      waitings_array,
                                                      cases):
                    trace = list(trace)
                    duration = list(durations)
                    waiting = list(waitings)
                    try:
                        index = trace.index(parms['ac_index']['end'])+1
                    except ValueError:
                        index = parms.get('max_trace_size')
                    for task, dur, wait in zip(trace[:index], duration[:index], waiting[:index]):
                        generated_event_log.append({'caseid': case, 'task': task, 'dur': dur, 'wait': wait})
                return decode_traces(parms, generated_event_log)
            except Exception:
                traceback.print_exc()
        return gen(*args)


def main(argv):
    def catch_parameter(opt):
        """Change the captured parameters names"""
        switch = {'-h': 'help', '-f': 'folder'}
        return switch.get(opt)
    param = dict()
    param['rep'] = 5
    # arg_max, random
    param['variant'] = 'random'
    # param settled manually or catched by console for batch operations
    if not argv:
        # Event-log filename
        param['folder'] = '20210211_D14093EF_146D_4F09_84C4_0A1EE83B0E75'
    else:
        # Catch parms by console
        try:
            opts, _ = getopt.getopt(argv, "h:f:", ['folder='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if key in ['max_eval']:
                    param[key] = int(arg)
                else:
                    param[key] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
    
    sim_values = list()
    # Generate logs
    generator = GanEventLogGenerator(param)
    param['file_name'] = generator.parms['file_name']
    for run_num in range(0, param['rep']):
        event_log = generator.generate()
        # export predictions
        export_predictions(run_num, event_log, param)
        # assesment
        sim_values.extend(
            evaluate_predict_log(param, generator.log,
                                 event_log, run_num))
    # Export results
    pd.DataFrame(sim_values).to_csv(
        os.path.join('output_files', 
                     param['folder'], 
                     sup.file_id(prefix='SE_')), 
        index=False)

def export_predictions(r_num, event_log, param):
    output_folder = os.path.join('output_files', param['folder'])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    event_log.to_csv(
        os.path.join(
            output_folder, 'gen_'+ 
            param['file_name'].split('.')[0]+'_'+str(r_num+1)+'.csv'), 
        index=False)

def evaluate_predict_log(parms, log, sim_log, rep_num):
    """Reads the simulation results stats
    Args:
        settings (dict): Path to jar and file names
        rep (int): repetition number
    """
    sim_values = list()
    log = copy.deepcopy(log)
    log = log[~log.task.isin(['Start', 'End', 'start', 'end'])]
    log['caseid'] = log['caseid'].astype(str)
    parms['read_options'] = dict()
    parms['read_options']['one_timestamp'] = False
    sim_log = sim_log[~sim_log.task.isin(['Start', 'End', 'start', 'end'])]
    evaluator = ev.SimilarityEvaluator(log, sim_log, parms)
    metrics = ['tsd', 'day_hour_emd', 'log_mae', 'dl', 'mae']
    for metric in metrics:
        evaluator.measure_distance(metric)
        sim_values.append({**{'run_num': rep_num}, **evaluator.similarity})
    return sim_values


if __name__ == "__main__":
    main(sys.argv[1:])
