"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains: automated tool to update the Copyright notice in the docstrings of dvas files
"""

import argparse
from pathlib import Path


def copyright_notice(years):
    """ Return the copyright notice.

    Args:
        years (str): the copyright years.

    Returns:
        str : the copyright notice.
    """

    notice = 'Copyright (c) {} MeteoSwiss, contributors listed in AUTHORS.\n'.format(years)

    return notice


def main():
    """ The main function that updates the Copyright notices of all .py files in the dvas
    repository.

    This is designed to be used through the command line as::

        python update_copyright.py 2020-2021

    """

    # Get the dvas version
    with open(Path('..') / 'src' / 'dvas' / 'version.py') as fid:
        version = next(line.split("'")[1] for line in fid.readlines() if 'VERSION' in line)

    # Use argparse to make dvas user friendly
    parser = argparse.ArgumentParser(
        description='DVAS {}'.format(version) +
        ' - Data Visualization and Analysis Software: Copyright update routine for devs.',
        epilog='For more info: https://MeteoSwiss.github.io/dvas\n ',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('years', action='store', default='2020-2022',
                        # choices=DVAS_RECIPES,
                        help='The years during which dvas was modified.')

    # Done getting ready. Now start doing stuff.
    # What did the user type in ?
    args = parser.parse_args()

    new_cn = copyright_notice(args.years)

    print('\n New copyright notice:\n')
    print(new_cn)

    answer = None
    while answer not in ['y', 'n']:
        answer = input("Proceed [y/n]: ")
    print(' ')

    if answer == 'n':
        print('Aborting ...')
        return None

    # Get a list of all the .py files in the repository
    fn_list = Path('..').rglob('*.py')

    # Very well, loop through these, and update them as I see fit.
    for f_name in fn_list:

        f_object = open(f_name, 'r')
        content = f_object.readlines()
        f_object.close()

        copyright_line = [l for (l, line) in enumerate(content) if 'Copyright (c)' in line]

        if len(copyright_line) == 0:
            print('Ignoring {}: no pre-existing copyright info ...'.format(f_name))
            continue
        if len(copyright_line) > 1:
            print('Ignoring {}: too many copyright occurences ...'.format(f_name))
            continue

        # Replace the line
        content[copyright_line[0]] = new_cn

        # Write the update file
        f_object = open(f_name, 'w')
        f_object.writelines(content)
        f_object.close()

    print('\nAll .py files updated.\n')
    print('Do not forget to update the following files manually:\n')
    print('   ./.dev_utils/update_copyright.py')
    print('   ./docs/source/substitutions.rst')
    print('   ./docs/source/conf.py')
    print('   ./github/workflows/CI_xxx.yml')
    print(' ')


# Make everything above actually work.
if __name__ == "__main__":
    main()
