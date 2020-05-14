'''
This script will check the files modified in a given pull_request, to see if a specific one was
modified. Raises an Exception if not.

Inspired by:
https://stackoverflow.com/questions/25071579/list-all-files-changed-in-a-pull-request-in-git-github
https://stackoverflow.com/questions/11113896/use-git-commands-within-python-code

Created May 2020; F.P.A. Vogt; frederic.vogt@alumni.anu.edu.au
'''

import argparse
import os
import subprocess

def main():
    ''' The main function. '''

    # Use argparse to allow to feed parameters to this script
    parser = argparse.ArgumentParser(description='''Checks if a specific file has been modified in a
                                                    pull_request. Intended to be used with a
                                                    Github Action.''',
                                     epilog='Feedback, questions, comments: \
                                             frederic.vogt@alumni.anu.edu.au \n',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--filename', action='store', metavar='file_name',
                        default=None, help='''Name of file to check.''')

    # What did the user type in ?
    args = parser.parse_args()

    if args.filename is None:
        raise Exception('Ouch! Please specify a filename!')

    # Use git to find all the modified files
    pipe = subprocess.PIPE

    process = subprocess.Popen(['git', '--no-pager', 'diff', '--name-only', 'FETCH_HEAD...'],
                               stdout=pipe, stderr=pipe)
    stdoutput, _ = process.communicate()

    # Decode the bytes
    stdoutput = stdoutput.decode("utf-8")

    # Check each file in this commit.
    for item in [os.path.basename(file) for file in stdoutput.split('\n')]:
        if item == args.filename:
            print('%s was modified in this pull_request' % (args.filename))
            return True

    # If I get to this point, then the file was not changed. Raise an Exception about it.
    raise Exception('Ouch ! %s was not modified in this pull_request.' % (args.filename))

if __name__ == '__main__':

    main()
