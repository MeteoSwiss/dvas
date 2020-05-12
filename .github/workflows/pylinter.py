# -*- coding: utf-8 -*-
'''
This script can be used together with a Github Action to run pylint on all the .py files in a
repository. Command line arguments can be used to search for a specific subset of errors (which,
if found, will raise an Exception), or to ignore some in a generic search (which will print all the
error found, but will not raise any Exception).

Created May 2020; F.P.A. Vogt; frederic.vogt@alumni.anu.edu.au

'''

import argparse
import glob
import os
from pylint import epylint as lint

def main():
    ''' The main function. '''

    # Use argparse to allow to feed parameters to this script
    parser = argparse.ArgumentParser(description='''Runs pylint on all .py files in a folder and all
                                                    its subfolders. Intended to be used with a
                                                    Github Action.''',
                                     epilog='Feedback, questions, comments: \
                                             frederic.vogt@alumni.anu.edu.au \n',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--restrict', action='store', metavar='error codes', nargs='+',
                        default=None, help='''List of space-separated error codes to strictly
                                            restrict the search for.''')

    parser.add_argument('--exclude', action='store', metavar='error codes', nargs='+',
                        default=None, help='List of space-separated error codes to ignore.')

    # What did the user type in ?
    args = parser.parse_args()

    # Do I want to only run pylint for a specific few errors ONLY ?
    if args.restrict is not None:

        error_codes = ','.join(args.restrict)
        pylint_command = '--disable=all --enable='+error_codes

    # or do I rather want to simply exclude some errors ?
    elif args.exclude is not None:
        error_codes = ','.join(args.exclude)
        pylint_command = '--disable='+error_codes

    else: # just run pylint without tweaks

        pylint_command = ''

    # Get a list of all the .py files here and in all the subfolders.
    fn_list = ' '.join(glob.glob(os.path.join('.', '**', '*.py'), recursive=True))

    # Launch pylint with the appropriate options
    (pylint_stdout, _) = lint.py_run(fn_list + ' ' + pylint_command, return_std=True)

    # Extract the score
    score = pylint_stdout.getvalue().split('\n')[-3].split('rated at ')[1].split('/10 ')[0]

    # For the Github Action, raise an exception in case I get any restricted errors.
    if args.restrict is not None and score != '10':
        # Display the output, so we can learn somehting from it if needed
        print(pylint_stdout.getvalue())
        raise Exception('Ouch! Some forbidden pylint error codes are present!')

    # If I do not have any restricted errors, then simply show the pylint errors without failing.
    print(pylint_stdout.getvalue())

if __name__ == '__main__':

    main()
