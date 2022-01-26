from rdkit import Chem
import os
import shutil
import subprocess

import numpy as np

from .g16_log import XtbLog


def xtb_optimization(folder, sdf, xtb_path, logger):
    basename = os.path.basename(sdf)
    file_name = os.path.splitext(basename)[0]
    scratch_dir = os.path.join(folder,file_name)
    os.makedirs(scratch_dir)

    pwd = os.getcwd()

    os.chdir(scratch_dir)

    try:
        xtb_command = os.path.join(xtb_path, 'xtb')
        with open(os.path.join('..','{}_xtb_opt.log'.format(file_name)), 'w') as out:
            print(xtb_command, os.path.join('..','{}.sdf'.format(file_name)))
            subprocess.call([xtb_command, os.path.join('..','{}.sdf'.format(file_name)), '-opt'],
                            stdout=out, stderr=out)
            shutil.move('xtbopt.sdf', os.path.join('..','{}_opt.sdf'.format(file_name)))
            os.remove(os.path.join('..','{}.sdf'.format(file_name)))

        with open(os.path.join('..',file_name + '_freq.log'), 'w') as out:
            subprocess.call([xtb_command, os.path.join('..','{}_opt.sdf'.format(file_name)), '-ohess'], stdout=out,
                            stderr=out)

            # os.remove('hessian')
            # os.remove('vibspectrum')

        log = XtbLog(os.path.join('..','{}_freq.log'.format(file_name)))
    finally:
        os.chdir(pwd)
        shutil.rmtree(scratch_dir)

    if log.termination:
        peaks = log.wavenum
        if np.min(peaks) < 0:
            raise RuntimeError('imaginary frequency found for {}'.format(file_name))
        else:
            return '{}_opt.sdf'.format(file_name)
    else:
        raise RuntimeError('xtb optimization did not finish for {}'.format(file_name))
