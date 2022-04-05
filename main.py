from argparse import ArgumentParser, Namespace
import os
import shutil
import time
import yaml

import pandas as pd

import rdkit.Chem as Chem

from lib import create_logger
from lib import csearch
from lib import xtb_optimization
from lib import dft_scf

XTB_PATH = '/home/gridsan/oscarwu/bin/anaconda3/envs/QM_descriptors/bin/'
G16_PATH = '/home/gridsan/oscarwu/GRPAPI/Software/g16/'

parser = ArgumentParser()
parser.add_argument('--ismiles', type=str, required=False,
                    help='input smiles included in a .csv file')
# parser.add_argument('--output', type=str, default='QM_descriptors.pickle',
#                     help='output as a .pickle file')
# conformer searching
parser.add_argument('--MMFF_conf_folder', type=str, default='MMFF_conf',
                    help='folder for MMFF searched conformers')
parser.add_argument('--nconf', type=int, default=500,
                    help='number of MMFF conformers')
parser.add_argument('-max_conf_try', type=int, default=2000,
                    help='maximum attempt for conformer generating, '
                         'this is useful for molecules with many chiral centers.')
parser.add_argument('-rmspre', type=float, required=False,
                        help='rms threshold pre optimization')
parser.add_argument('--rmspost', type=float, required=False, default=0.4,
                    help='rms threshold post MMFF minimization')
parser.add_argument('--E_cutoff', type=float, required=False, default=10.0,
                    help='energy window for MMFF minimization')
parser.add_argument('--MMFF_threads', type=int, required=False, default=40,
                    help='number of process for the MMFF conformer searching')
parser.add_argument('--timeout', required=False, default=600,
                    help='time window for each MMFF conformer searching sub process')
# xtb optimization
parser.add_argument('--xtb_folder', type=str, default='XTB_opt',
                    help='folder for XTB optimization')

# DFT calculation
parser.add_argument('--DFT_folder', type=str, default='DFT',
                    help='folder for DFT calculation')
parser.add_argument('--DFT_theory', type=str, default='b3lyp/def2svp',
                    help='level of theory for the DFT calculation')
parser.add_argument('--DFT_n_procs', type=int, default=4,
                    help='number of process for DFT calculations')
parser.add_argument('--DFT_job_ram', type=int, default=3000,
                    help='amount of ram (MB) allocated for each DFT calculation')
# Split number
parser.add_argument('--split', type=int, default=None,
                        help='split number for multi-part job')

# Job control
parser.add_argument('--only_DFT', action='store_true', help='only perform DFT related jobs')


args = parser.parse_args()

name = os.path.splitext(args.ismiles)[0]
logger = create_logger(name=name)

df = pd.read_csv(args.ismiles, index_col=0)
# create id to smile mapping
molid_to_smi_dict = dict(zip(df.id, df.smiles))
molid_to_charge_dict = dict()
for k, v in molid_to_smi_dict.items():
    try:
        mol = Chem.MolFromSmiles(v)
    except Exception as e:
        logger.error(f'Cannot translate smi {v} to molecule for species {k}')

    try:
        charge = Chem.GetFormalCharge(mol)
        molid_to_charge_dict[k] = charge
    except Exception as e:
        logger.error(f'Cannot determine molecular charge for species {k} with smi {v}')

# conformer searching

if not args.only_DFT:
    logger.info('starting MMFF conformer searching')
    supp = (x for x in df[['id', 'smiles']].values)
    conf_sdfs = csearch(supp, len(df), args, logger)

# xtb optimization

if not args.only_DFT:
    logger.info('starting GFN2-XTB structure optimization for the lowest MMFF conformer')
    os.makedirs(args.xtb_folder,exist_ok=True)

    opt_sdfs = []
    for conf_sdf in conf_sdfs:
        try:
            shutil.copyfile(os.path.join(args.MMFF_conf_folder, conf_sdf),
                            os.path.join(args.xtb_folder, conf_sdf))
            opt_sdf = xtb_optimization(args.xtb_folder, conf_sdf, XTB_PATH, logger)
            opt_sdfs.append(opt_sdf)
        except Exception as e:
            logger.error('XTB optimization for {} failed: {}'.format(os.path.splitext(conf_sdf)[0], e))

# G16 DFT calculation
if not args.only_DFT:
    os.makedirs(args.DFT_folder, exist_ok=True)
else:
    logger.info("Searching for optimized XTB files.")
    opt_sdfs = []
    for a_file in os.listdir(args.DFT_folder):
        if a_file.endswith(".sdf"):
            logger.info(f'Found file {a_file}')
            molid = a_file.split('_')[0]
            if molid in molid_to_smi_dict.keys():
                opt_sdfs.append(a_file)

qm_descriptors = dict()
for opt_sdf in opt_sdfs:
    try:
        molid = opt_sdf.split('_')[0]
        charge = molid_to_charge_dict[molid]
    except Exception as e:
        logger.error(f'Cannot determine molecular charge for species {molid}')

    if not args.only_DFT:
        try:
            shutil.copyfile(os.path.join(args.xtb_folder, opt_sdf),
                            os.path.join(args.DFT_folder, opt_sdf))
            time.sleep(1)
        except Exception as e:
            logger.error(f'file IO error.')

    try:
        qm_descriptor = dft_scf(args.DFT_folder, opt_sdf, G16_PATH, args.DFT_theory, args.DFT_n_procs,
                                logger, args.DFT_job_ram, charge)
    except Exception as e:
        logger.error('Gaussian optimization for {} failed: {}'.format(os.path.splitext(opt_sdf)[0], e))

    try:
        molid = opt_sdf.split('_')[0]
        smi = molid_to_smi_dict[molid]
        qm_descriptors[molid] = (smi, qm_descriptor)
    except Exception as e:
        logger.error(f'descriptor store error main.py line 143 - 144')

if args.split is None:
    with open('qm_descriptors.yaml', 'w') as output:
        yaml.dump(qm_descriptors, output)
else:
    with open(f'qm_descriptors_{args.split}.yaml', 'w') as output:
        yaml.dump(qm_descriptors, output)


