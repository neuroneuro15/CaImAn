# -*- coding: utf-8 -*-
""" functions related to the creation and management of the cluster

This file contains

We put arrays on disk as raw bytes, extending along the first dimension.
Alongside each array x we ensure the value x.dtype which stores the string
Description h

See Also:
------------

@url
.. image::
@author andrea giovannucci
"""
# \package caiman
# \version   1.0
# \copyright GNU General Public License v2.0
# \date Created on Thu Oct 20 12:07:09 2016

from __future__ import print_function

import subprocess
import time
import ipyparallel
from ipyparallel import Client
import shutil
import glob
import shlex
import psutil
import sys
import os
import numpy as np
from multiprocessing import Pool            
import multiprocessing


def start_server(slurm_script=None, ipcluster="ipcluster", ncpus = None):
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
    sys.stdout.write("Starting cluster...")
    sys.stdout.flush()
    if ncpus is None:
        ncpus=psutil.cpu_count()

    if slurm_script is None:
        if ipcluster == "ipcluster":
            subprocess.Popen("ipcluster start -n {0}".format(ncpus), shell=True, close_fds=(os.name != 'nt'))
        else:
            subprocess.Popen(shlex.split("{0} start -n {1}".format(ipcluster, ncpus)), shell=True, close_fds=(os.name != 'nt'))

        # Check that all processes have started
        time.sleep(1)
        client = ipyparallel.Client()
        while len(client) < ncpus:
            sys.stdout.write(".")
            sys.stdout.flush()
            client.close()

            time.sleep(1)
            client = ipyparallel.Client()
        client.close()

    else:
        shell_source(slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)
        ee = c[:]
        ne = len(ee)
        print(('Running on %d engines.' % (ne)))
        c.close()
        sys.stdout.write(" done\n")


def shell_source(script):
    """Sometime you want to emulate the action of "source"

    in bash, settings some environment variables. Here is a way to do it.
    """
    pipe = subprocess.Popen(". %s; env" % script,  stdout=subprocess.PIPE, shell=True)
    output = pipe.communicate()[0]
    env = dict()
    for line in output.splitlines():
        lsp = line.split("=", 1)
        if len(lsp) > 1:
            env[lsp[0]] = lsp[1]
    os.environ.update(env)
    pipe.stdout.close()


def stop_server(ipcluster='ipcluster',pdir=None,profile=None, dview = None):
    """
    programmatically stops the ipyparallel server

    Parameters:
     ----------
     ipcluster : str
         ipcluster binary file name; requires 4 path separators on Windows
         Default: "ipcluster"

    """
    if 'multiprocessing' in str(type(dview)):
        dview.terminate()
    else:
        sys.stdout.write("Stopping cluster...\n")
        sys.stdout.flush()
        try:
            pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
            is_slurm = True
        except:
            print('NOT SLURM')
            is_slurm = False
    
        if is_slurm:
            if pdir is None and profile is None:
                pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
            c = Client(ipython_dir=pdir, profile=profile)
            ee = c[:]
            ne = len(ee)
            print(('Shutting down %d engines.' % (ne)))
            c.close()
            c.shutdown(hub=True)
            shutil.rmtree('profile_' + str(profile))
            try:
                shutil.rmtree('./log/')
            except:
                print('creating log folder')
    
            files = glob.glob('*.log')
            os.mkdir('./log')
    
            for fl in files:
                shutil.move(fl, './log/')
    
        else:
            if ipcluster == "ipcluster":
                proc = subprocess.Popen("ipcluster stop", shell=True, stderr=subprocess.PIPE, close_fds=(os.name != 'nt'))
            else:
                proc = subprocess.Popen(shlex.split(ipcluster + " stop"),
                                        shell=True, stderr=subprocess.PIPE, close_fds=(os.name != 'nt'))
    
            line_out = proc.stderr.readline()
            if b'CRITICAL' in line_out:
                sys.stdout.write("No cluster to stop...")
                sys.stdout.flush()
            elif b'Stopping' in line_out:
                st = time.time()
                sys.stdout.write('Waiting for cluster to stop...')
                while (time.time() - st) < 4:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    time.sleep(1)
            else:
                print(line_out)
                print('**** Unrecognized Syntax in ipcluster output, waiting for server to stop anyways ****')
    
            proc.stderr.close()

    sys.stdout.write(" done\n")


def setup_cluster(backend = 'multiprocessing',n_processes = None,single_thread = False):
    """ Restart if necessary the pipyparallel cluster, and manages the case of SLURM

    Parameters:
    ----------
    backend: str
        'multiprocessing', 'ipyparallel', and 'SLURM'
    """
    #todo: todocument

    if n_processes is None:
        if backend == 'SLURM':
            n_processes = np.int(os.environ.get('SLURM_NPROCS'))
        else:
            # roughly number of cores on your machine minus 1
            n_processes = np.maximum(np.int(psutil.cpu_count()), 1)

    if single_thread:
        dview = None
        c = None
    else:
        sys.stdout.flush()

        if backend == 'SLURM':
            try:
                stop_server(is_slurm=True)
            except:
                print('Nothing to stop')
            slurm_script = 'SLURM/slurmStart.sh'
            start_server(slurm_script=slurm_script, ncpus=n_processes)
            pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
            c = Client(ipython_dir=pdir, profile=profile)
        elif backend is 'ipyparallel':
            stop_server()
            start_server(ncpus=n_processes)
            c = Client()
            print(('Using ' + str(len(c)) + ' processes'))
            dview = c[:len(c)]

        elif (backend is 'multiprocessing') or (backend is 'local'):
            if len (multiprocessing.active_children())>0:
                raise Exception('A cluster is already runnning. Terminate with dview.terminate() if you want to restart.')
            c = None
            dview = Pool(n_processes)
        else:
            raise Exception('Unknown Backend')
        
    return c,dview,n_processes
