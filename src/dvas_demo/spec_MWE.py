""" This is here only for a short time, until issue #121 gets dealt with. """

import sys

if __name__ == '__main__':

    print(dir(sys.modules['__main__']))

    if '__spec__' in dir(sys.modules['__main__']):
        print('__spec__ is None:', sys.modules['__main__'].__spec__ is None)
