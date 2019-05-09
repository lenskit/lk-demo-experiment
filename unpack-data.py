"""
Unpack a data set.

Usage:
  unpack-data.py <zip-file> [<dest-dir>]
"""

import zipfile

from docopt import docopt

args = docopt(__doc__)

zfname = args['<zip-file>']
dir = args.get('<dest-dir>', None)
if dir is None:
  dir = '.'

with zipfile.ZipFile(zfname) as zf:
    for member in zf.namelist():
        print('extracting', member)
        zf.extract(member, path=dir)