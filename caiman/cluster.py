# -*- coding: utf-8 -*-
""" functions related to the creation and management of the cluster

This file contains

We put arrays on disk as raw bytes, extending along the first dimension.
Alongside each array x we ensure the value x.dtype which stores the data type.

@author andrea giovannucci
"""
# \package caiman
# \version   1.0
# \copyright GNU General Public License v2.0
# \date Created on Thu Oct 20 12:07:09 2016

from __future__ import print_function, division, absolute_import

import sys
import os
import subprocess
import time
import glob
import shlex
import psutil
import shutil
from multiprocessing import Pool
import multiprocessing
import ipyparallel
import numpy as np


slurm_script = 'SLURM/slurmStart.sh'


def shell_source(script):
    """ Run a source-style bash script, copy resulting env vars to current process. """
    with subprocess.Popen(". %s; env" % script, stdout=subprocess.PIPE, shell=True) as pipe:
        output = pipe.communicate()[0]
        env = dict([line.split('=', 1) for line in str(output).splitlines() if '=' in line])
        os.environ.update(bytes(env))


class Cluster(object):

    def __init__(self, backend='multiprocessing', n_cpus=psutil.cpu_count(), n_processes=0, startup_script='', ipcluster='ipcluster'):

        if backend.lower() not in ['slurm', 'multiprocessing', 'ipyparallel', 'local']:
            raise ValueError("backend must be either 'slurm', 'multiprocessing', 'ipyparallel', or 'local'.")
        if backend.lower() == 'slurm' and not startup_script:
            raise ValueError("a slurm_script must be supplied if using a slurm backend.")

        self.backend = backend.lower()
        self.startup_script = startup_script
        self._n_cpus = int(n_cpus)
        self.setup()
        self.n_processes = n_processes
        self.ipcluster = ipcluster

        self.client, self.dview = self.setup()

    def setup(self):
        # todo: todocument

        if self.backend in ['slurm', 'ipyparallel']:
            self.stop_server()
            self.start_server()
            client = ipyparallel.Client(ipython_dir=self.pdir, profile=self.profile)
            dview = client[:len(client)]  # is this correct for 'slurm' backend (original code resulted in NameError.)?
        else:
            client = None
            dview = Pool(self.n_processes)

        return client, dview

    @property
    def is_running(self):
        if self.backend == 'slurm':
            pass
        elif self.backend in ['multiprocessing', 'local']:
            return len(multiprocessing.active_children()) > 0

    @property
    def n_cpus(self):
        return self._n_cpus

    @property
    def n_processes(self):
        if not self._n_processes:
            if self.backend == 'slurm':
                return np.int(os.environ.get('SLURM_NPROCS'))
            else:
                return max(psutil.cpu_count(), 1)

    @n_processes.setter
    def n_processes(self, value):
        self._n_processes = int(value)

    @property
    def pdir(self):
        return os.environ['IPPPDIR'] if self.backend == 'slurm' else None

    @property
    def profile(self):
        return os.environ['IPPPROFILE'] if self.backend == 'slurm' else None

    def start_server(self, ipcluster="ipcluster"):
        """
        programmatically start the ipyparallel server

        Parameters:
        ----------
        ncpus: int
            number of processors

        ipcluster : str
            ipcluster binary file name; requires 4 path separators on Windows. ipcluster="C:\\\\Anaconda2\\\\Scripts\\\\ipcluster.exe"
             Default: "ipcluster"
        """
        print("Starting cluster...", end='\r')

        if self.backend == 'slurm':
            shell_source(self.startup_script)
        else:
            subprocess.Popen(shlex.split("{0} start -n {1}".format(ipcluster, self.n_cpus)), shell=True, close_fds=(os.name != 'nt'))
            time.sleep(1)

        # Check that all processes have started
        client = ipyparallel.Client(ipython_dir=self.pdir, profile=self.profile)
        client.close()

    def stop_server(self):
        """
        programmatically stops the ipyparallel server

        Parameters:
         ----------
         ipcluster : str
             ipcluster binary file name; requires 4 path separators on Windows
             Default: "ipcluster"

        """
        print("Stopping cluster...\n", end='\r')

        if 'multiprocessing' in str(type(self.dview)):
            self.dview.terminate()
            return

        if self.backend == 'slurm':
            c = ipyparallel.Client(ipython_dir=self.pdir, profile=self.profile)
            c.close()
            c.shutdown(hub=True)
            shutil.rmtree('profile_' + self.profile)
            self.write_log()
        else:
            with subprocess.Popen(['ipcluster', 'stop'], shell=True, stderr=subprocess.PIPE, close_fds=(os.name != 'nt')) as proc:
                line_out = proc.stderr.readline()
                print(line_out)
                if b'Stopping' in line_out:
                    time.sleep(4)

    def write_log(self):
        shutil.rmtree('./log/', ignore_errors=True)
        os.mkdir('./log')
        for fl in glob.glob('*.log'):
            shutil.move(fl, './log/')