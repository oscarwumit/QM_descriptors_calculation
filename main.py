from argparse import ArgumentParser
import os
import shutil
import json
import pandas as pd

from lib import create_logger
from lib import csearch
from lib import xtb_optimization
from lib import dft_scf
from lib import G16NBOQMData
from lib import json2csv

XTB_PATH = '$GFN_XTB_PATH'
G16_PATH = '$G16_PATH'

def main():
    """
    This script performs quantum mechanical (QM) calculations and collects QM descriptors for a set of molecules.
    It takes input in the form of a .csv file containing SMILES strings of the molecules.
    The script performs the following steps:
    1. Conformer searching using MMFF force field.
    2. XTB optimization of the lowest energy conformer obtained from MMFF search.
    3. DFT calculation using Gaussian16 for each optimized conformer.
    4. Collection of QM descriptors from the DFT calculation results.
    The QM descriptors are saved in separate files for each molecule in the specified output folder.
    """

    parser = ArgumentParser()

    parser.add_argument('--ismiles', type=str, required=False,
                        help='input smiles included in a .csv file')

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
    parser.add_argument('--DFT_n_procs', type=int, default=20,
                        help='number of process for DFT calculations')

    # QM descriptors parsing
    parser.add_argument('--QM_des_folder', type=str, default='QM_descriptors',
                        help='folder for saving QM descriptors')

    args = parser.parse_args()

    name = os.path.splitext(args.ismiles)[0]
    logger = create_logger(name=name)

    df = pd.read_csv(args.ismiles, index_col=0)

    # conformer searching
    logger.info('starting MMFF conformer searching')
    supp = (x for x in df[['id', 'smiles']].values)
    conf_sdfs = csearch(supp, len(df), args, logger)

    # xtb optimization
    logger.info('starting GFN2-XTB structure optimization for the lowest MMFF conformer')
    os.makedirs(args.xtb_folder, exist_ok=True)

    opt_sdfs = []
    for conf_sdf in conf_sdfs:
        try:
            shutil.copyfile(os.path.join(args.MMFF_conf_folder, conf_sdf),
                            os.path.join(args.xtb_folder, conf_sdf))
            opt_sdf = xtb_optimization(args.xtb_folder, conf_sdf, XTB_PATH, logger)
            opt_sdfs.append(opt_sdf)
        except Exception as e:
            logger.error('XTB optimization for {} failed: {}'.format(
                os.path.splitext(conf_sdf)[0], e))

    # G16 DFT calculation
    os.makedirs(args.DFT_folder, exist_ok=True)

    for opt_sdf in opt_sdfs:
        try:
            shutil.copyfile(os.path.join(args.xtb_folder, opt_sdf),
                            os.path.join(args.DFT_folder, opt_sdf))
            dft_scf(args.DFT_folder, opt_sdf, G16_PATH, args.DFT_theory, args.DFT_n_procs,
                    logger)
        except Exception as e:
            logger.error('Gaussian optimization for {} failed: {}'.format(
                os.path.splitext(opt_sdf)[0], e))

    # QM descriptors parsing
    os.makedirs(args.QM_des_folder, exist_ok=True)

    base_log_dir = os.path.join(args.DFT_folder, "base")
    plus_log_dir = os.path.join(args.DFT_folder, "plus")
    minus_log_dir = os.path.join(args.DFT_folder, "minus")

    base_qm_descriptors, plus_qm_descriptors, minus_qm_descriptors = {}, {}, {}
    df = pd.read_csv(args.ismiles, index_col=0)
    for id, smi in df[['id', 'smiles']].values:
        base_log = os.path.join(base_log_dir, f'{id}_opt.log')
        plus_log = os.path.join(plus_log_dir, f'{id}_opt.log')
        minus_log = os.path.join(minus_log_dir, f'{id}_opt.log')
        if os.path.exists(base_log) and os.path.exists(plus_log) and os.path.exists(minus_log):
            base_qm = G16NBOQMData(logFilePath=base_log,
                                identifier=id, smi=smi, dataType='base')
            plus_qm = G16NBOQMData(logFilePath=plus_log,
                                identifier=id, smi=smi, dataType='plus')
            minus_qm = G16NBOQMData(logFilePath=minus_log,
                                identifier=id, smi=smi, dataType='minus')
            base_qm_descriptors[id] = base_qm.data
            plus_qm_descriptors[id] = plus_qm.data
            minus_qm_descriptors[id] = minus_qm.data

    for jobtype, db in [('base', base_qm_descriptors), ('plus', plus_qm_descriptors), ('minus', minus_qm_descriptors)]:
        output_path = os.path.join(args.QM_des_folder, f'qm_{jobtype}.json')
        with open(output_path, 'w') as json_file:
            json.dump(db, json_file)

    base_path = os.path.join(args.QM_des_folder, f'qm_base.json')
    plus_path = os.path.join(args.QM_des_folder, f'qm_plus.json')
    minus_path = os.path.join(args.QM_des_folder, f'qm_minus.json')
    j2c = json2csv(base_path, plus_path, minus_path)
    output_path = os.path.join(args.QM_des_folder, "qm_des.csv")
    j2c.save_descriptors(output_path)

if __name__ == "__main__":
    main()
