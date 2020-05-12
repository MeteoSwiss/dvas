# -*- coding: utf-8 -*-
'''
This script can be used together with a Github Action to run pylint on all the .py files in a
repository.

Created May 2020; F.P.A. Vogt; frederic.vogt@alumni.anu.edu.au

'''

import argparse

if __name__== '__main__':

    # Use argparse to allow to feed parameters to this script
    parser = argparse.ArgumentParser(description='''Runs pylint on all .py files in a folder and all
                                                    its subfolders. Intended to be used with a 
                                                    Github Action.''',
                                 epilog='Feedback, questions, comments: \
                                         frederic.vogt@alumni.anu.edu.au \n',
                                 formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--restrict', action='store', metavar='error codes', nargs='+',
                    default=None, help='List of space-separated error codes to strictly restrict '+
                                       'the search for.')

    parser.add_argument('--exclude', action='store', metavar='error codes', nargs='+',
                    default=None, help='List of space-separated error codes to ignore.')

    # What did the user type in ?
    args = parser.parse_args()

    if args.restrict is None:
        raise Exception('Ouch!')

    print('Error code(s): ', args.restrict)    