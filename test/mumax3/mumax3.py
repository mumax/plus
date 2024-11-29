from hashlib import sha1
import os
import shutil
import subprocess
import sys

import numpy as np
from pandas import read_table

CACHEDIR = sys.path[0] + "/.mumax3_cache"


def clear_mumax3_cache():
    shutil.rmtree(CACHEDIR)


def remove_white_space(string):
    lines = string.split("\n")
    lines = [line.strip() for line in lines if line.strip() != ""]
    string = "\n".join(lines)
    return string


class Mumax3Simulation:
    def __init__(self, script):
        self._script = remove_white_space(script)
        if not os.path.exists(self.outputdir):
            self.run()

        if os.path.exists(self.outputdir + "/table.txt"):
            self._table = read_table(self.outputdir + "/table.txt")
            self._table.columns = ' '.join(self._table.columns).split()[1::2]
        else:
            self._table = None

    @property
    def script(self):
        return self._script

    @property
    def hash(self):
        return sha1(self._script.encode()).hexdigest()

    @property
    def scriptfile(self):
        return CACHEDIR + "/" + self.hash + ".mx3"

    @property
    def outputdir(self):
        return CACHEDIR + "/" + self.hash + ".out"

    @property
    def table(self):
        """Returns a pandas dataframe of the mumax³ table, or None if no table
        was found."""
        return self._table

    def run(self):
        if not os.path.exists(CACHEDIR):
            os.makedirs(CACHEDIR)

        if os.path.exists(self.outputdir):
            shutil.rmtree(self.outputdir)

        with open(self.scriptfile, "w") as f:
            f.write(self.script)

        subprocess.run(
            ["mumax3", self.scriptfile], check=True, stdout=open(os.devnull, "wb")
        )

        subprocess.run(
            ["mumax3-convert", "-numpy", self.outputdir + "/*.ovf"],
            check=True,
            stdout=open(os.devnull, "wb"),
            stderr=open(os.devnull, "wb"),
        )

    def get_field(self, fieldname):
        filename = self.outputdir + "/" + fieldname + ".npy"
        return np.load(filename)

    def get_column(self, columnname):
        return self.table[columnname]
