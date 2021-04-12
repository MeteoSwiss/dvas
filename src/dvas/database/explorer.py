"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Local database exploring tools

"""

# Import from python packages
from cmd import Cmd
from shlex import split
from argparse import ArgumentParser
from pprint import pprint, pformat

# Import from current package
from .database import DatabaseManager
from .model import Parameter, Info
from .search import SearchInfoExpr, AllExpr
from ..errors import SearchError


class Parser(ArgumentParser):

    def exit(self, status, message):
        print(message)

    def error(self, message):
        print(message)


class DatabasePrompt(Cmd):
    """"""

    prompt = 'db> '
    intro = "Welcome! Type '?' to list commands"

    _EXIT_CMD = ['exit', 'quit', 'x', 'q']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Init attributes
        self._db_mngr = DatabaseManager()
        self._db_reader = ReadDatabase()

    def default(self, inp):
        if inp in self._EXIT_CMD:
            return self.do_exit(inp)

    def do_exit(self, _):
        pprint("Bye! Bye!")
        return True

    def help_exit(self):
        pprint(f"Type ({'|'.join(self._EXIT_CMD)}) to exit")

    def do_prm(self, inp):

        try:
            pprint(self._db_reader.prm(inp))

        except Exception as exc:
            pprint(exc)

    def help_prm(self):
        pprint(self._db_reader.prm('?'))

    def do_info(self, inp):

        try:
            pprint(self._db_reader.info(inp))

        except Exception as exc:
            pprint(exc)

    def help_info(self):
        pprint(self._db_reader.info('?'))

    def do_obj(self, inp):

        try:
            pprint(self._db_reader.obj(inp))

        except Exception as exc:
            pprint(exc)

    def help_obj(self):
        pprint(self._db_reader.obj('?'))


class ReadDatabase:
    """Class used to display DB content"""

    def __init__(self):

        # Init attributes
        self._db_mngr = DatabaseManager()
        self._parser = Parser()

        # Set parser
        self._parser.add_argument('expr', default=None, nargs='?')
        self._parser.add_argument('--length', '-l', action='store_true', default=False)
        self._parser.add_argument('--recurse', '-r', action='store_true', default=False)

    def _execute(self, search_type, search_expr):
        """Execute method

        Args:
            search_type (str): 'info' | 'prm' | 'obj'
            search_expr (str): Search expression

        """

        # Init
        SearchInfoExpr.set_stgy(search_type)
        self._parser.prog = search_type

        # Split
        search_expr = split(search_expr)

        # Parse
        try:

            # Check help '?'
            if search_expr == ['?']:
                self._parser.parse_args(['-h'])

            args = self._parser.parse_args(search_expr)

            # Define
            expr = args.expr
            length = args.length
            recurse = args.recurse

            # Get all if None
            if expr is None:
                expr = AllExpr()

            # Search in DB
            out = SearchInfoExpr.eval(expr, out='dict', recurse=recurse)

            if length:
                out = len(out)

        except SystemExit:
            out = None

        return out

    def info(self, expr):
        """Search from Info table"""
        return self._execute('info', expr)

    def prm(self, expr):
        """Search from Parameter table"""
        return self._execute('prm', expr)

    def obj(self, expr):
        """Search from Object table"""
        return self._execute('obj', expr)
