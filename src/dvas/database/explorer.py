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


class DatabasePrompt(Cmd):
    """"""

    prompt = 'db> '
    intro = "Welcome! Type '?' to list commands"

    _EXIT_CMD = ['exit', 'x', 'q']

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

        #try:
        out = self._db_reader.prm(inp)

        # except Exception as exc:
        #     out = exc.args[0]

        pprint(out)

    def help_prm(self):
        pprint("Display 'Parameter' table")

    def do_info(self, inp):

        try:
            out = self._db_reader.info(inp)

        except Exception as exc:
            out = exc.args[0]

        pprint(out)

    def help_info(self):
        pprint("Display 'Info' table")


class ReadDatabase:
    """Class used to display DB content"""

    def __init__(self):

        # Init attributes
        self._db_mngr = DatabaseManager()
        self._parser = ArgumentParser()

        # Set parser
        self._parser.add_argument('expr', default=None, nargs='?')
        self._parser.add_argument('--length', '-l', action='store_true', default=False)

    def info(self, expr):

        # Init
        SearchInfoExpr.set_stgy('info')
        self._parser.prog = 'info'

        # Parse
        try:
            args = self._parser.parse_args(split(expr))

        except Exception:
            raise Exception(f"Error in parsing '{expr}'")

        # Define
        expr = args.expr
        length = args.length

        # Get all if None
        if expr is None:
            expr = AllExpr()

        try:
            out = SearchInfoExpr.eval(expr, out='dict')

            if length:
                out = len(out)

        except Exception:
            raise Exception('Error in argument')

        return out

    def prm(self, expr):

        # Init
        SearchInfoExpr.set_stgy('prm')
        self._parser.prog = 'prm'

        # Parse
        #try:
        args = self._parser.parse_args(split(expr))

        # except Exception:
        #     raise Exception(f"Error in parsing '{expr}'")

        # Define
        expr = args.expr
        length = args.length

        # Get all if None
        if expr is None:
            expr = AllExpr()

        #try:
        out = SearchInfoExpr.eval(expr, out='dict')

        if length:
            out = len(out)

        # except Exception:
        #     raise Exception('Error in argument')

        return out
