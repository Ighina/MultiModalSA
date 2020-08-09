# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:41:20 2020

@author: Iacopo
"""

import os
import argparse
import shutil
import sys

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = MyParser(
        description='Workspace generator function: the program will create\
        a directory with a user-specified name in which a set of predefined sub-directories\
        will contain the training data, generated sentences, saved model, saved vocabularies\
        respectively. It is advised to use this function for each new experiment, so\
        as to create separate locations for each experiment.')

parser.add_argument('--directory_name', '-name', default='experiment', type=str,
                    help='Name of the directory to be created.')

parser.add_argument('--data_folder', '-data', default='data', type=str,
                    help='name of the subdirectory storing the data. If two or\
                    more experiments share the same hyperparameters but different\
                    data, a differently named data folder can be defined (in which\
                    case the corresponding line in json file needs to be changed)')


args = parser.parse_args()

os.mkdir(args.directory_name)

os.chdir(args.directory_name)

os.mkdir('saved_model')

os.mkdir('outputs')

os.mkdir('checkpoint')

shutil.copyfile('../Hyperparameters.json','Hyperparameters.json')

print('New directory set up at {}: copy the data in {}, change the parameters of Hyperparameters.json as required and you are ready to go!'.format(os.getcwd(),
      os.path.join(os.getcwd(),args.data_folder)))

    
