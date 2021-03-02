
import sys

from pathlib import Path



if __name__ == '__main__':

    print(dir(sys.modules['__main__']))
    #sys.modules['__main__'].__spec__ is None
