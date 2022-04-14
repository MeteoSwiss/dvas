"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Local database exploring tools

"""

# Import from python packages
from cmd import Cmd
from shlex import split
from argparse import ArgumentParser
from pprint import pprint
from inspect import stack
from re import match, sub

# Import from current package
from .database import DatabaseManager
from .search import SearchInfoExpr, AllExpr


class ParserExit(Exception):
    """Exception for parser exiting"""


class Parser(ArgumentParser):
    """Parser class"""

    def exit(self, status=0, message=None):
        """Overwrite exit method

        Notes:
            - Interrupt system exit behavior

        """
        raise ParserExit()

    def error(self, message):
        """Overwrite error method

        Notes:
            - Interrupt system exit behavior

        """
        print(message)


class DatabasePrompt(Cmd):
    """Class for DB prompt command interface"""

    prompt = 'db> '
    intro = "Welcome! Type '?' to list commands"

    _EXIT_CMD = ['exit', 'quit', 'x', 'q']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Init attributes
        self._db_mngr = DatabaseManager()
        self._db_reader = ReadDatabase()

    def default(self, inp):
        """Default commande method"""
        if inp in self._EXIT_CMD:
            return self.do_exit(inp)

    def _exec_cmd(self, arg=''):
        """Execute command

        Args:
            arg (str, `optional`): Command argument. Default to ''

        Note:
            - Call it only from a do_* or help_* method

        """

        # Get origin method name
        fct_name = stack()[1][3]
        cmd_name = sub(r'^((do)|(help))_(\w*)$', r'\4', fct_name)
        do_cmd = True if match('^do', fct_name) else False

        # Test
        assert (cmd_name != '') and (cmd_name != fct_name),\
            'Must be called from a do_* or help_* function'

        # Execute command
        if do_cmd:
            try:
                pprint(getattr(self._db_reader, cmd_name)(arg))

            except Exception as exc:
                pprint(exc)

        else:
            pprint(getattr(self._db_reader, cmd_name)('?'))

    def do_exit(self, _):
        """Exit command"""
        pprint("Bye! Bye!")
        return True

    def help_exit(self):
        """Exit command help"""
        pprint(f"Type ({'|'.join(self._EXIT_CMD)}) to exit")

    def do_prm(self, inp):
        """Parameter command"""
        self._exec_cmd(inp)

    def help_prm(self):
        """Parameter command help"""
        self._exec_cmd()

    def do_info(self, inp):
        """Info command"""
        self._exec_cmd(inp)

    def help_info(self):
        """Info command help"""
        self._exec_cmd()

    def do_obj(self, inp):
        """Object command"""
        self._exec_cmd(inp)

    def help_obj(self):
        """Object command help"""
        self._exec_cmd()


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
        search_expr = split(search_expr, posix=False)

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

        except ParserExit:
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
