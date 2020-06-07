"""
Unpack a data set.

Usage:
  unpack-data.py <zip-file> [<dest-dir>]
"""

import os, os.path
import zipfile

from lkdemo import log
from docopt import docopt

def main(args):
    zfname = args['<zip-file>']
    dir = args.get('<dest-dir>', None)
    if dir is None:
    dir = '.'
    if not os.path.exists(dir):
        os.mkdir(dir)

    with zipfile.ZipFile(zfname) as zf:
        for member in zf.namelist():
            print('extracting', member)
            zf.extract(member, path=dir)

if __name__ == '__main__':
    _log = log.script(__file__)
    args = docopt(__doc__)
    main(args)
