#!/usr/bin/env python

import glob
import os
import shutil

import setuptools

"""
files = None
tdir = None
if os.path.exists(tdir) is False:
    os.mkdir(tdir)
for file in files:
    dst = os.path.join(tdir, os.path.basename(file))
    shutil.copy(file, dst)
"""

if __name__ == "__main__":
    setuptools.setup(package_data={})
