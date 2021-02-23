# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 18:23:16 2021

@author: Manuel Camargo
"""
import os
import sys
import getopt

import torch
import utils.support as sup

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
import event_log_gen_input as ei
import event_timestamp_prediction as etp
    
def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-f': 'file_name', '-p': 'prefix'}
    return switch.get(opt)

def train_event_timestamp_gen(param):
    param['mode'] = 'event_two_timestamps'
    obj = ei.Input()
    obj.run(param)
    rnnG = etp.LSTMGenerator(seq_len = obj.prefix_len, 
                            input_size = len(obj.selected_columns), 
                            batch = obj.batch, 
                            hidden_size= 2*len(obj.selected_columns) , 
                            num_layers = 2, 
                            num_directions = 1)
    optimizerG = torch.optim.Adam(rnnG.parameters(), 
                                  lr=0.0002, 
                                  betas=(0.5, 0.999))

    #Initializing a discriminator
    rnnD = etp.LSTMDiscriminator(seq_len = obj.prefix_len+1, 
                                input_size = len(obj.selected_columns), 
                                batch = obj.batch, 
                                hidden_size = 2*len(obj.selected_columns), 
                                num_layers =2, 
                                num_directions = 1)
    optimizerD = torch.optim.Adam(rnnD.parameters(), 
                                  lr=0.0002, 
                                  betas=(0.5, 0.999))

    #Training and testing
    etp.train(rnnD=rnnD, 
              rnnG=rnnG,
              optimizerD=optimizerD, 
              optimizerG=optimizerG, 
              obj=obj, 
              epoch=param['epoch'])
    return obj.log_test


def main(argv):
    param = dict()
    column_names = {'Case ID': 'caseid',
                    'Activity': 'task',
                    'lifecycle:transition': 'event_type',
                    'Resource': 'user'}
    param['one_timestamp'] = False  # Only one timestamp in the log
    param['read_options'] = {
        'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
        'column_names': column_names,
        'one_timestamp': param['one_timestamp']}
    param['input_path'] = 'dataset'
    param['output_path'] = os.path.join('output_files', sup.folder_id())
    param['epoch'] = 25
    param['batch_size'] = 5
    # param settled manually or catched by console for batch operations
    if not argv:
        # Event-log filename
        param['file_name'] = 'PurchasingExample.xes'
        param['prefix'] = 4
    else:
        # Catch parms by console
        try:
            opts, _ = getopt.getopt(argv, "h:f:p:", ['file_name=', 'prefix='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if key in ['prefix']:
                    param[key] = int(arg)
                else:
                    param[key] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
            
    log_test = train_event_timestamp_gen(param)
    log_test.to_csv(os.path.join(param['output_path'], 'tst_'+
                                  param['file_name'].split('.')[0]+'.csv'),
                          index=False,
                          encoding='utf-8')
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
