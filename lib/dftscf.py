from rdkit import Chem
import os
import subprocess
from .file_parser import mol2xyz, xyz2com


def dft_scf(folder, sdf, g16_path, level_of_theory, n_procs, logger):
    basename = os.path.basename(sdf)

    parent_folder = os.getcwd()
    os.chdir(folder)

    try:
        file_name = os.path.splitext(basename)[0]

        mol = Chem.SDMolSupplier(sdf, removeHs=False, sanitize=False)[0]
        xyz = mol2xyz(mol)

        num_radical_elec = 0
        for atom in mol.GetAtoms():
            num_radical_elec += atom.GetNumRadicalElectrons()
        base_charge = Chem.GetFormalCharge(mol)
        base_mult = num_radical_elec + 1

        pwd = os.getcwd()

        g16_command = os.path.join(g16_path, 'g16')
        for jobtype in ['base', 'plus', 'minus']:
            os.makedirs(jobtype, exist_ok=True)

            if jobtype == 'base':
                charge = base_charge
                mult = base_mult
                head = '%chk={}.chk\n%nprocshared={}\n# {} nmr=GIAO scf=(maxcycle=512, xqc) ' \
                       'pop=(full,mbs,hirshfeld,nbo6read) iop(7/33=1) iop(2/9=2000)\n'.format(
                           file_name, n_procs, level_of_theory)
            elif jobtype == 'plus':
                charge = base_charge + 1
                mult = base_mult + 1
                head = '%chk={}.chk\n%nprocshared={}\n# {} scf=(maxcycle=512, xqc) ' \
                       'pop=(full,mbs,hirshfeld,nbo6read) iop(7/33=1) iop(2/9=2000)\n'.format(
                           file_name, n_procs, level_of_theory)
            elif jobtype == 'minus':
                charge = base_charge - 1
                mult = base_mult + 1
                head = '%chk={}.chk\n%nprocshared={}\n# {} scf=(maxcycle=512, xqc) ' \
                       'pop=(full,mbs,hirshfeld,nbo6read) iop(7/33=1) iop(2/9=2000)\n'.format(
                           file_name, n_procs, level_of_theory)

            os.chdir(jobtype)
            comfile = file_name + '.gjf'
            xyz2com(xyz, head=head, comfile=comfile, charge=charge,
                    mult=mult, footer='$NBO BNDIDX $END\n')

            logfile = file_name + '.log'
            outfile = file_name + '.out'
            with open(outfile, 'w') as out:
                subprocess.run('{} < {} >> {}'.format(
                    g16_command, comfile, logfile), shell=True, stdout=out, stderr=out)
            os.chdir(pwd)
        os.remove(sdf)
    finally:
        os.chdir(parent_folder)
