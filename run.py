# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:00:51 2020

@author: Iacopo
"""

import os
import json
import argparse
import re
import sys

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = MyParser(
        description = 'Run training with hyperparameters defined in the relative json file')

parser.add_argument('--experiment_folder', '-folder', default='experiment', type=str,
                    help='Folder storing results and containing data for the current experiment.')

parser.add_argument('--hyper_file', '-hyper', default='Hyperparameters.json', type=str, 
                    help='Configuration file defining the hyperparameters and options to be used in training.')

args = parser.parse_args()

hyper_file = os.path.join(args.experiment_folder, args.hyper_file)

assert os.path.exists(hyper_file), "Configuration file wasn't detected in the directory from which you are\
 running the current script: please move the configuration file to this directory --> {}".format(os.getcwd())


with open(hyper_file, encoding='utf-8') as f:
    temp = f.read()

hyper = json.loads(temp)

param = ["python Train.py --experiment_folder {}".format(args.experiment_folder)]

for key,value in hyper.items():
    if re.findall("TRUE",str(value)):
        param.append(str(key))
    elif re.findall("FALSE", str(value)):
        pass
    else:
        param.append(str(key)+' '+str(value))
        
command = " ".join(param)


def main():
    os.system(command)
#    print(command)
if __name__=='__main__':
    main()
