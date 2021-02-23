# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:04:49 2021

@author: Manuel Camargo
"""
import os
import time
import utils.support as sup


# =============================================================================
#  Support
# =============================================================================

def create_file_list(path):
    file_list = list()
    for root, dirs, files in os.walk(path):
        for f in files:
            file_list.append(f)
    return file_list

# =============================================================================
# Sbatch files creator
# =============================================================================


def sbatch_creator(log, prefix):
    exp_name = (os.path.splitext(log)[0]
                    .lower()
                    .split(' ')[0][:5])
    if imp == 2:
        default = ['#!/bin/bash',
                   '#SBATCH --partition=gpu',
                   '#SBATCH --gres=gpu:tesla:1',
                   '#SBATCH -J ' + exp_name,
                   '#SBATCH -N 1',
                   '#SBATCH --mem=32000',
                   '#SBATCH -t 120:00:00',
                   'module load python/3.6.3/virtenv',
                   'source activate sc_gan_taymouri',
                   ]
    else:
        default = ['#!/bin/bash',
                   '#SBATCH --partition=main',
                   '#SBATCH -J '+exp_name,
                   '#SBATCH -N 1',
                   '#SBATCH --cpus-per-task=10',
                   '#SBATCH --mem=32000',
                   '#SBATCH -t 120:00:00',
                   'module load python/3.6.3/virtenv',
                   'source activate sc_gan_taymouri',
                   ]
    options = 'python event_log_training.py -f ' + log
    options += ' -p ' + str(prefix)
    default.append(options)
    file_name = sup.folder_id()
    sup.create_text_file(default, os.path.join(output_folder, file_name))

# =============================================================================
# Sbatch files submission
# =============================================================================

def sbatch_submit(in_batch, bsize=20):
    file_list = create_file_list(output_folder)
    print('Number of experiments:', len(file_list), sep=' ')
    for i, _ in enumerate(file_list):
        if in_batch:
            if (i % bsize) == 0:
                time.sleep(20)
                os.system('sbatch '+os.path.join(output_folder, file_list[i]))
            else:
                os.system('sbatch '+os.path.join(output_folder, file_list[i]))
        else:
            os.system('sbatch '+os.path.join(output_folder, file_list[i]))

# =============================================================================
# Kernel
# =============================================================================


# create output folder
output_folder = 'jobs_files'
# Xserver ip

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# clean folder
for _, _, files in os.walk(output_folder):
    for file in files:
        os.unlink(os.path.join(output_folder, file))

# parameters definition
imp = 1  # keras lstm implementation 1 cpu, 2 gpu
logs = [
        'BPI_Challenge_2012_W_Two_TS.xes',
        'BPI_Challenge_2017_W_Two_TS.xes',
        'poc_processmining.xes',
        'PurchasingExample.xes',
        'Production.xes',
        'ConsultaDataMining201618.xes',
        'insurance.xes',
        'callcentre.xes',
        ]

for log in logs:
    for pref in [5, 10 , 15]:
        # sbatch creation
        sbatch_creator(log, pref)
# submission
sbatch_submit(False)
