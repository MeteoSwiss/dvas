
from cmd import Cmd
from pprint import pprint

from .database import DatabaseManager
from .model import Parameter, Info


class DatabasePrompt(Cmd):
    prompt = 'db> '
    intro = "Welcome! Type ? to list commands"

    _EXIT_CMD = ['x', 'q']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Init attributes
        self._db_mngr = DatabaseManager()

    def default(self, inp):
        if inp in self._EXIT_CMD:
            return self.do_exit(inp)

    def do_exit(self, inp):
        pprint("Bye! Bye!")
        return True

    def do_prm(self, inp):
        pprint(self._db_mngr.get_table(Parameter))

    def help_prm(self):
        pprint("Display 'Parameter' table")

    def do_info(self, inp):
        pprint(self._db_mngr.get_table(Info))

    def help_info(self):
        pprint("Display 'Info' table")
