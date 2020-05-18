# -*- coding: utf-8 -*-
'''
This script can be used together with a Github Action to run pylint on all the .py files in a
repository. Command line arguments can be used to search for a specific subset of errors (if any are
found, this script will raise an Exception), or to ignore some errors in a generic search (which
will print all the errors found, but will not raise any Exception). If a score is specified, the
script will raise an Exception if it is not met.

Created May 2020; F.P.A. Vogt; frederic.vogt@alumni.anu.edu.au
'''

import argparse
import glob
import os
import re
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

    parser.add_argument('--min_score', type=float, action='store', metavar='float<10', default=None,
                        help='Minimum acceptable score, below which an Exception should be raised')

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

    # Extract the score ... keep it as an int for now.
    # score = round(float(
    #     pylint_stdout.getvalue().split('\n')[-3].split('rated at ')[1].split('/10 ')[0]
    # ), 2)
    score = round(float(
        re.search(r'(-?\d+(\.\d+)?)/10', pylint_stdout.getvalue())[1]
    ), 2)

    # For the Github Action, raise an exception in case I get any restricted errors.
    if args.restrict is not None and score < 9.99:
        # Display the output, so we can learn something from it if needed
        print(pylint_stdout.getvalue())
        raise Exception('Ouch! Some forbidden pylint error codes are present!')

    # If I do not have any restricted errors, then simply show the pylint errors without failing.
    print(pylint_stdout.getvalue())

    # If a minimum score was set, raise an Exception if it is not met, so that it can be picked-up
    # by a Github Action.
    if args.min_score is not None:
        if score < args.min_score:
            raise Exception('''Ouch! pylint final score of %.2f is smaller than the specified
                               threshold of %.2f !''' % (float(score), args.min_score))

if __name__ == '__main__':

    main()
