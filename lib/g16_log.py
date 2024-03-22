import os
import re
import sys
import pandas as pd
import numpy as np
import math
from rdmc.external.gaussian import GaussianLog
from cclib.method import CSPA


class G16NBOQMData:

    periodictable = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
                     "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
                     "Kr", "Rb", "Sr", "Y", "Zr",
                     "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
                     "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
                     "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
                     "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
                     "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Uub", "Uut", "Uuq",
                     "Uup", "Uuh", "Uus", "Uuo"]

    def __init__(self, logFilePath, identifier=None, smi='', dataType=''):
        self.file = logFilePath
        self.basename = os.path.basename(logFilePath)

        self.id = identifier
        self.smi = smi
        self.data_type = dataType

        self.GetRDMCLog()

        if self.success:
            self.ReadLogFile()
            self._CheckOpenShell()
            self._SplitNBOLogOpenShell()
            self.GetChargeMultiplicity()
            self.GetXYZ()
            self.GetBondLengthMatrix()
            self.GetCPU()
            self.GetAtoms()
            self.GetG16SCF()
            self.GetMullikenCharge()
            self.GetMullikenDipole()
            self.GetMullikenDipoleQuadrupoleTraceless()
            self.GetMullikenCondensedChargeMatrix()
            self.GetHirshfeld()
            self.GetNPACharge()
            self.GetNAOValenceElecConfig()
            self.GetNAOFullPopulation()
            self.GetNPAWibergBondIdxMatrix()
            self.GetNPAWibergBondIdxbyAtom()
            self.GetNaturalBindingIdxMatrix()
            self.GetNBOLewis()
            self.GetNBOLewisFullSet()
            self.GetNBOLewisEnergy()
            self.GetNLMOBondOrderMatrix()
            self.GetNLMOBondOrderMatrixbyAtom()
            self.GetNBONMR()
            self.SCFOrbitalEnergy()
            self.GetCSPA()
            self.GetDataDict()

    @staticmethod
    def elementID(massno):
        if massno < len(G16NBOQMData.periodictable):
            return G16NBOQMData.periodictable[massno]
        else:
            return "XX"

    def GetRDMCLog(self):
        self.glog = GaussianLog(self.file)
        self.success = self.glog.success

    def ReadLogFile(self):
        with open(self.file) as fh:
            txt = fh.readlines()
        self.log = tuple([x.strip() for x in txt])

    def _CheckOpenShell(self):
        for i, line in enumerate(self.log):
            if line.find('alpha spin orbitals') > -1:
                self.open_shell = True
                break
        else:
            self.open_shell = False

    def GetChargeMultiplicity(self):
        try:
            self.charge = self.glog.charge
        except ValueError:
            with open(self.file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if "charge" in line.lower() and "multiplicity" in line.lower():
                        charge = int(line.split()[2])
                        return charge
                raise ValueError("Charge is not found...")
        self.multiplicity = self.glog.multiplicity

    def GetXYZ(self):
        xyz = self.glog.cclib_results.writexyz(indices=0)
        self.xyz = "\n".join([l for l in xyz.splitlines()[2:]])

    def GetMullikenCondensedChargeMatrix(self):
        n_atoms = len(self.atom_dict)
        with open(self.file, 'r') as f:
            line = f.readline()
            while line != '':
                if "    Condensed to atoms (all electrons)" in line:
                    condesed_atomic_charges = np.zeros(
                        (n_atoms, n_atoms), np.float64)
                    for i in range(int(math.ceil(n_atoms / 6.0))):
                        # Header row
                        line = f.readline()
                        # Matrix element rows
                        for j in range(n_atoms):
                            data = f.readline().split()
                            for k in range(len(data) - 2):
                                condesed_atomic_charges[j, i *
                                                        6 + k] = float(data[k + 2])
                                condesed_atomic_charges[i * 6 + k,
                                                        j] = condesed_atomic_charges[j, i * 6 + k]
                line = f.readline()
        self.mulliken_condensed_charge_matrix = condesed_atomic_charges.tolist()

    def GetBondLengthMatrix(self):
        m = re.findall(r"[+-]?\d+\.\d*[Ee]?[+-]?\d*", self.xyz)
        xyz = [float(x) for x in m]
        natom = int(len(xyz)/3)
        tmp = [[] for i in range(natom)]
        for i in range(natom):
            for j in range(3):
                tmp[i].append(xyz[3*i+j])
        bond_length_matrix = [
            ['e' for j in range(natom)] for i in range(natom)]
        for i in range(natom):
            for j in range(i+1):
                if i == j:
                    bond_length_matrix[i][j] = 0.0
                delta_x = tmp[i][0] - tmp[j][0]
                delta_y = tmp[i][1] - tmp[j][1]
                delta_z = tmp[i][2] - tmp[j][2]
                bond_length = (delta_x ** 2 + delta_y **
                               2 + delta_z ** 2) ** 0.5
                bond_length_matrix[i][j] = bond_length
                bond_length_matrix[j][i] = bond_length
        self.bond_length_matrix = tuple(
            [tuple([float(x) for x in k]) for k in bond_length_matrix])

    def GetCPU(self):
        for line in (self.log):
            if line.find("Job cpu time") > -1:
                days = int(line.split()[3])
                hours = int(line.split()[5])
                mins = int(line.split()[7])
                secs = float(line.split()[9])
                self.CPU = tuple([days, hours, mins, secs])
                break

    def GetAtoms(self):
        starting = False
        found_coord = False
        for line in (self.log):
            if line.find('orientation') > -1:
                starting = True
                AtomsNum = []
                AtomsType = []
                Coords = []
                sep = 0
                found_coord = False
            if starting:
                m = re.search('(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)',
                              line)
                if not m:
                    continue
                AtomsNum.append(int(m.group(2)))
                found_coord = True
            if found_coord and line.find('-----------') > -1:
                starting = False
                found_coord = False

        atoms = tuple([self.elementID(x) for x in AtomsNum])
        self.atom_dict = {i+1: a for i, a in enumerate(atoms)}

    def GetG16SCF(self):
        for line in (self.log):
            if line.find("SCF Done:") > -1:
                self.method = line.split()[2]
                self.energy = float(line.split()[4])
                break

    def GetMullikenCharge(self):
        txt = self.log
        for i, line in enumerate(txt):
            if line.find('Sum of Mulliken charges') > -1:
                txt = txt[:i]
                break

        for i, line in enumerate(txt):
            if line.find('Mulliken charges') > -1:
                txt = txt[i:]
                break

        mulliken_charge = list()
        mulliken_spin = list()

        if 'spin' not in txt[0]:
            # neutral or base log
            for i, line in enumerate(txt[2:]):
                mulliken_charge.append(float(line.split()[2]))
                mulliken_spin.append(None)
        else:
            # plus or minus log
            for i, line in enumerate(txt[2:]):
                mulliken_charge.append(float(line.split()[2]))
                mulliken_spin.append(float(line.split()[3]))

        self.mulliken_charge = tuple(mulliken_charge)
        self.mulliken_spin = tuple(mulliken_spin)

    def GetMullikenDipole(self):
        txt = (x.strip() for x in self.log)
        dipole_moment = ''
        while True:
            try:
                line = next(txt)
            except StopIteration as e:
                break

            if re.match('Dipole moment', line):
                line = next(txt)
                m = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
                dipole_moment = m
                break

        mulliken_dipole_moment = tuple([float(x) for x in dipole_moment])
        self.mulliken_dipole_x = mulliken_dipole_moment[0]
        self.mulliken_dipole_y = mulliken_dipole_moment[1]
        self.mulliken_dipole_z = mulliken_dipole_moment[2]
        self.mulliken_dipole_tot = mulliken_dipole_moment[3]

    def GetMullikenDipoleQuadrupoleTraceless(self):
        txt = (x.strip() for x in self.log)

        dipole_moment = []
        while True:
            try:
                line = next(txt)
            except StopIteration as e:
                break

            if re.match('Traceless Quadrupole moment', line):
                line = next(txt)
                m = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
                dipole_moment.append(tuple([float(x) for x in m]))
                line = next(txt)
                m = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
                dipole_moment.append(tuple([float(x) for x in m]))
                break

        mulliken_dipole_moment = tuple(dipole_moment)
        self.mulliken_dipole_xx = mulliken_dipole_moment[0][0]
        self.mulliken_dipole_yy = mulliken_dipole_moment[0][1]
        self.mulliken_dipole_zz = mulliken_dipole_moment[0][2]

        self.mulliken_dipole_xy = mulliken_dipole_moment[1][0]
        self.mulliken_dipole_xz = mulliken_dipole_moment[1][1]
        self.mulliken_dipole_yz = mulliken_dipole_moment[1][2]

        # https://onlinelibrary.wiley.com/doi/10.1002/jcc.21417
        self.mulliken_quadrupoles = (
            2*(self.mulliken_dipole_xx**2 + self.mulliken_dipole_yy**2 + self.mulliken_dipole_zz**2)/3)**0.5

    def GetHirshfeld(self):
        txt = self.log
        for i, line in enumerate(txt):
            if line.find('Hirshfeld charges, spin densities, dipoles, and CM5 charges') > -1:
                txt = txt[i+2:]

        hirshfeld_charges = []
        hirshfeld_spin_density = []
        hirshfeld_dipoles_x = []
        hirshfeld_dipoles_y = []
        hirshfeld_dipoles_z = []
        hirshfeld_cm5_charges = []
        for i, line in enumerate(txt):
            if line.find('Tot') > -1:
                break
            m = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
            if m:
                hirshfeld_charges.append(m[1])
                hirshfeld_spin_density.append(m[2])
                hirshfeld_dipoles_x.append(m[3])
                hirshfeld_dipoles_y.append(m[4])
                hirshfeld_dipoles_z.append(m[5])
                hirshfeld_cm5_charges.append(m[6])

        self.hirshfeld_charges = tuple([float(x) for x in hirshfeld_charges])
        self.hirshfeld_spin_density = tuple(
            [float(x) for x in hirshfeld_spin_density])
        self.hirshfeld_dipoles_x = tuple(
            [float(x) for x in hirshfeld_dipoles_x])
        self.hirshfeld_dipoles_y = tuple(
            [float(x) for x in hirshfeld_dipoles_y])
        self.hirshfeld_dipoles_z = tuple(
            [float(x) for x in hirshfeld_dipoles_z])
        self.hirshfeld_cm5_charges = tuple(
            [float(x) for x in hirshfeld_cm5_charges])

    def _SplitNBOLogOpenShell(self):
        txt = self.log

        if self.open_shell:
            for i, line in enumerate(txt):
                if line.find('Perform NBO analysis') > -1:
                    nbo_idx = i
                    break

            for i, line in enumerate(txt):
                if line.find('Alpha spin orbitals') > -1:
                    alpha_idx = i
                    break

            for i, line in enumerate(txt):
                if line.find('Beta  spin orbitals') > -1:
                    beta_idx = i
                    break

            self.txt_overall = tuple(txt[nbo_idx:alpha_idx])
            self.txt_alpha = tuple(txt[alpha_idx:beta_idx])
            self.txt_beta = tuple(txt[beta_idx:])

    @staticmethod
    def _GetNPACharge(txt):
        NPA_Charge = list()

        for i, line in enumerate(txt):
            if line.find('Summary of Natural Population Analysis') > -1:
                head = txt[i + 4]
                txt = txt[i + 6:]
                break

        for i, line in enumerate(txt):
            if line.find('====================') > -1:
                txt = txt[:i]
                break

        for i, line in enumerate(txt):
            m = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
            atom = re.findall("[a-zA-Z]+", line)[0]
            atom_idx = int(m[0])
            natural_charge = float(m[1])
            cor_pop = float(m[2])
            val_pop = float(m[3])
            ryd_pop = float(m[4])
            tot_pop = float(m[5])

            if 'Density' not in head:
                natural_spin_density = None
            else:
                natural_spin_density = float(m[6])

            NPA_Charge.append(tuple(
                [atom_idx, atom, natural_charge, cor_pop, val_pop, ryd_pop, tot_pop, natural_spin_density]))
        return tuple(NPA_Charge)

    def GetNPACharge(self):
        if self.open_shell:
            self.npa_charge_open_shell_overall = self._GetNPACharge(
                self.txt_overall)
            self.npa_charge_alpha_spin_orbital = self._GetNPACharge(
                self.txt_alpha)
            self.npa_charge_beta_spin_orbital = self._GetNPACharge(
                self.txt_beta)
            self.npa_charge_closed_shell_overall = None
            self.npa_charge = [i[2]
                               for i in self.npa_charge_open_shell_overall]
            self.npa_spin_density = [i[-1]
                                     for i in self.npa_charge_open_shell_overall]
        else:
            self.npa_charge_open_shell_overall = None
            self.npa_charge_alpha_spin_orbital = None
            self.npa_charge_beta_spin_orbital = None
            self.npa_charge_closed_shell_overall = self._GetNPACharge(self.log)
            self.npa_charge = [i[2]
                               for i in self.npa_charge_closed_shell_overall]
            self.npa_spin_density = None

        self.npa_charge_meta = tuple(['atom_idx', 'atom', 'natural_charge',
                                     'cor_pop', 'val_pop', 'ryd_pop', 'tot_pop', 'natural_spin_density'])

    @staticmethod
    def _GetElecConfig(txt):

        electron_configuration = []

        for i, line in enumerate(txt):
            if line.find('Natural Electron Configuration') > -1:
                txt = txt[i + 2:]
                break

        for i, line in enumerate(txt):

            if not line:
                break

            if '[core]' in line:
                electron_configuration.append(
                    ''.join(line.split('[core]')[-1].split()))
            else:
                electron_configuration.append(''.join(line.split(re.findall(
                    r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)[0])[-1].split()))

        return tuple(electron_configuration)

    def GetNAOValenceElecConfig(self):

        if self.open_shell:
            self.elec_config_open_shell_overall = self._GetElecConfig(
                self.txt_overall)
            self.elec_config_alpha_spin_orbital = self._GetElecConfig(
                self.txt_alpha)
            self.elec_config_beta_spin_orbital = self._GetElecConfig(
                self.txt_beta)
            self.elec_config_closed_shell_overall = None
        else:
            self.elec_config_open_shell_overall = None
            self.elec_config_alpha_spin_orbital = None
            self.elec_config_beta_spin_orbital = None
            self.elec_config_closed_shell_overall = self._GetElecConfig(
                self.log)

    @staticmethod
    def _GetNAOFullPopulation(txt):

        for i, line in enumerate(txt):
            if line.find('NATURAL POPULATIONS:  Natural atomic orbital occupancies') > -1:
                start_idx = i
                break

        for i, line in enumerate(txt):
            if line.find('Summary of Natural Population Analysis:') > -1:
                end_idx = i
                break

        for i, line in enumerate(txt[start_idx:end_idx]):
            if line.find('NAO') > -1:
                title = line
                break

        txt = txt[start_idx + 4:end_idx - 2]

        for i, line in enumerate(txt):
            if line.find('effective core') > -1:
                cut_idx = i
                txt = txt[:cut_idx - 1]
                break

        NAO_population = list()

        for i, line in enumerate(txt):
            if not line:
                continue

            if "Val" in line:
                valency = 'Val'
            elif "Cor" in line:
                valency = 'Cor'
            elif "Ryd" in line:
                valency = 'Ryd'

            info = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
            info2 = re.findall("[a-zA-Z]+", line)

            nao_idx = int(info[0])
            element = info2[0]
            atom_order = int(info[1])
            principla_qm_number = line.split(')')[0].split('(')[-1].lstrip()[0]
            orbital = line.split(')')[0].split('(')[-1].lstrip()[-1]
            specific_orbital = line.split(valency)[0].split(
                str(atom_order))[-1].strip()

            orbital_notation = principla_qm_number + orbital
            specific_orbital_notation = principla_qm_number + specific_orbital

            occupancy = float(info[-2])

            if 'Energy' in title:
                energy = float(info[-1])
                spin = None
            elif 'Spin' in title:
                energy = None
                spin = float(info[-1])
            else:
                raise ValueError('Parser error.')

            NAO_population.append(tuple([nao_idx, atom_order, element, valency,
                                  orbital_notation, specific_orbital_notation, occupancy, energy, spin]))
        return tuple(NAO_population)

    def GetNAOFullPopulation(self):

        if self.open_shell:
            self.nao_pop_open_shell_overall = self._GetNAOFullPopulation(
                self.txt_overall)
            self.nao_pop_alpha_spin_orbital = self._GetNAOFullPopulation(
                self.txt_alpha)
            self.nao_pop_beta_spin_orbital = self._GetNAOFullPopulation(
                self.txt_beta)
            self.nao_pop_closed_shell_overall = None
        else:
            self.nao_pop_open_shell_overall = None
            self.nao_pop_alpha_spin_orbital = None
            self.nao_pop_beta_spin_orbital = None
            self.nao_pop_closed_shell_overall = self._GetNAOFullPopulation(
                self.log)

        self.nao_pop_meta = tuple(['nao_idx', 'atom_order', 'element', 'valency',
                                  'orbital_notation', 'specific_orbital_notation', 'occupancy', 'energy', 'spin'])

        n_atoms = len(self.atom_dict)
        self._1s_val, self._2s_val, self._2p_val, self._3s_val, self._3p_val, self._4s_val, self._4p_val = [
            [0] * n_atoms for i in range(7)]
        overall = self.nao_pop_open_shell_overall if self.open_shell else self.nao_pop_closed_shell_overall
        for line in overall:
            if line[3] == "Val":
                AO = line[4]
                atom_ind = line[1]
                occ = line[6]
                if AO == "1s":
                    self._1s_val[atom_ind-1] += occ
                elif AO == "2s":
                    self._2s_val[atom_ind-1] += occ
                elif AO == "2p":
                    self._2p_val[atom_ind-1] += occ
                elif AO == "3s":
                    self._3s_val[atom_ind-1] += occ
                elif AO == "3p":
                    self._3p_val[atom_ind-1] += occ
                elif AO == "4s":
                    self._4s_val[atom_ind-1] += occ
                elif AO == "4p":
                    self._4p_val[atom_ind-1] += occ
                else:
                    raise ValueError("Unrecognized orbital!")

    @staticmethod
    def _parseLargeMatrix(txt):

        data = dict()

        for i, line in enumerate(txt):
            if not line:
                continue

            try:
                _idx = line.split()[0][:-1]
            except IndexError:
                continue

            if not _idx.isnumeric():
                del _idx
                continue
            else:
                atom_idx = int(_idx)
                if atom_idx not in data.keys():
                    data[atom_idx] = list()

                data[atom_idx].extend([float(x) for x in line.split()[2:]])

        width = len(data[1])
        height = len(data.keys())

        if not width == height:
            raise ValueError('inconsistent matrix dimension, parser error')

        result = list()
        for i in range(width):
            atom_idx = i + 1
            result.append(tuple(data[atom_idx]))

        return tuple(result)

    @staticmethod
    def _GetNPAWibergBondIdxMatrix(txt):

        for i, line in enumerate(txt):
            if line.find('Wiberg bond index matrix in the NAO basis') > -1:
                txt = txt[i + 4:]
                break

        for i, line in enumerate(txt):
            if line.find('Wiberg bond index, Totals') > -1:
                txt = txt[:i]
                break

        bond_index_matrix = G16NBOQMData._parseLargeMatrix(txt)

        return bond_index_matrix

    def GetNPAWibergBondIdxMatrix(self):
        if self.open_shell:
            self.npa_wiberg_bdx_open_shell_overall = self._GetNPAWibergBondIdxMatrix(
                self.txt_overall)
            self.npa_wiberg_bdx_alpha_spin_orbital = self._GetNPAWibergBondIdxMatrix(
                self.txt_alpha)
            self.npa_wiberg_bdx_beta_spin_orbital = self._GetNPAWibergBondIdxMatrix(
                self.txt_beta)
            self.npa_wiberg_bdx_closed_shell_overall = None
        else:
            self.npa_wiberg_bdx_open_shell_overall = None
            self.npa_wiberg_bdx_alpha_spin_orbital = None
            self.npa_wiberg_bdx_beta_spin_orbital = None
            self.npa_wiberg_bdx_closed_shell_overall = self._GetNPAWibergBondIdxMatrix(
                self.log)

    @staticmethod
    def _GetNPAWibergBondIdxbyAtom(txt):
        bond_index_matrix = list()

        for i, line in enumerate(txt):
            if line.find('Wiberg bond index, Totals by atom:') > -1:
                txt = txt[i + 4:]
                break

        for i, line in enumerate(txt):
            if not line:
                break

            bond_index_matrix.append(float(line.split()[2]))

        return tuple(bond_index_matrix)

    def GetNPAWibergBondIdxbyAtom(self):
        if self.open_shell:
            self.npa_wiberg_bdx_by_atom_open_shell_overall = self._GetNPAWibergBondIdxbyAtom(
                self.txt_overall)
            self.npa_wiberg_bdx_by_atom_alpha_spin_orbital = self._GetNPAWibergBondIdxbyAtom(
                self.txt_alpha)
            self.npa_wiberg_bdx_by_atom_beta_spin_orbital = self._GetNPAWibergBondIdxbyAtom(
                self.txt_beta)
            self.npa_wiberg_bdx_by_atom_closed_shell_overall = None
        else:
            self.npa_wiberg_bdx_by_atom_open_shell_overall = None
            self.npa_wiberg_bdx_by_atom_alpha_spin_orbital = None
            self.npa_wiberg_bdx_by_atom_beta_spin_orbital = None
            self.npa_wiberg_bdx_by_atom_closed_shell_overall = self._GetNPAWibergBondIdxbyAtom(
                self.log)

    @staticmethod
    def _GetNaturalBindingIdxMatrix(txt):
        bond_index_matrix = list()

        for i, line in enumerate(txt):
            if line.find('NBI: Natural Binding Index') > -1:
                txt = txt[i + 4:]
                break

        for i, line in enumerate(txt):
            if line.find('NATURAL BOND ORBITAL ANALYSIS') > -1 or line.find('********') > -1:
                txt = txt[:i]
                break

        bond_index_matrix = G16NBOQMData._parseLargeMatrix(txt)

        return bond_index_matrix

    def GetNaturalBindingIdxMatrix(self):
        if self.open_shell:
            self.nbi_open_shell_overall = self._GetNaturalBindingIdxMatrix(
                self.txt_overall)
            self.nbi_alpha_spin_orbital = self._GetNaturalBindingIdxMatrix(
                self.txt_alpha)
            self.nbi_beta_spin_orbital = self._GetNaturalBindingIdxMatrix(
                self.txt_beta)
            self.nbi_closed_shell_overall = None
        else:
            self.nbi_open_shell_overall = None
            self.nbi_alpha_spin_orbital = None
            self.nbi_beta_spin_orbital = None
            self.nbi_closed_shell_overall = self._GetNaturalBindingIdxMatrix(
                self.log)

    @staticmethod
    def _GetNBOLewis(txt):
        for i, line in enumerate(txt):
            if line.find('NATURAL BOND ORBITAL ANALYSIS') > -1:
                txt = txt[i + 4:]
                break

        for i, line in enumerate(txt):
            if line.find('accepted') > -1:
                txt = txt[:i-2]

                if line.find('delocalized') > -1:
                    delocalized = 1
                elif line.find('low occupancy') > -1:
                    delocalized = 0
                else:
                    raise ValueError(f'Unexpected txt {txt[i]}')
                break

        data = txt[-1].split()
        lewis_occupancy = float(data[3])
        non_lewis_occupancy = float(data[4])
        lewis_core = float(data[5])
        lewis_two_center_bond = float(data[6])
        lewis_multi_center = float(data[7])
        lewis_lone_pair = float(data[8])
        low_occupancy_lewis = float(data[9])
        high_occupany_lewis = float(data[10])

        composite = tuple([lewis_occupancy, non_lewis_occupancy, lewis_core, lewis_two_center_bond,
                          lewis_multi_center, lewis_lone_pair, low_occupancy_lewis, high_occupany_lewis, delocalized])

        return composite

    def GetNBOLewis(self):
        if self.open_shell:
            self.nbo_lewis_open_shell_overall = None
            self.nbo_lewis_alpha_spin_orbital = self._GetNBOLewis(
                self.txt_alpha)
            self.nbo_lewis_beta_spin_orbital = self._GetNBOLewis(self.txt_beta)
            self.nbo_lewis_closed_shell_overall = None
        else:
            self.nbo_lewis_open_shell_overall = None
            self.nbo_lewis_alpha_spin_orbital = None
            self.nbo_lewis_beta_spin_orbital = None
            self.nbo_lewis_closed_shell_overall = self._GetNBOLewis(self.log)

        self.nbo_lewis_meta = tuple(['lewis_occupancy', 'non_lewis_occupancy', 'lewis_core', 'lewis_two_center_bond',
                                    'lewis_multi_center', 'lewis_lone_pair', 'low_occupancy_lewis', 'high_occupany_lewis', 'delocalized'])

    @staticmethod
    def _hybridization_calculator(nbo_hybrid):
        """
        Function to convert nbo hybridization to standard spd exponential format.

        Examples:

            s(100.00%) -> s = 1, p = 0, d = 0
            s(25.00%)p3.00(74.95%)d0.00(0.05%) -> s = 1, p = 3, d = 0
            s(0.00%)p1.00(99.89%)d0.00(0.11%) -> s = 0, p = 1, d = 0
            s(0.00%)p0.00(0.00%)d1.00(100.00%) -> s = 0, p = 0, d = 1
            s(0.00%)p1.00(33.33%)d2.00(66.67%) -> s = 0, p = 1, d = 2
            s(0.02%)p99.99(99.98%)d0.16(0.00%) -> s = 0, p = 1, d = 0
            s(0.08%)p99.99(99.84%)d1.05(0.08%) -> s = 0, p = 1, d = 0
            s(4.06%)p21.01(85.32%)d2.61(10.62%) -> s = 1, p = 21.01, d = 2.61
            s(0.44%)p20.86(9.10%)d99.99(90.47%) -> s = 0, p = 1, d = 5

            return all float 
        """

        orbitals = re.findall("[a-zA-Z]+", nbo_hybrid)
        if 'f' in orbitals:
            raise ValueError('Not implemented for f orbitals.')

        if len(orbitals) == 3 and 'd' in orbitals:
            result = G16NBOQMData._hybridization_calculator_spd(nbo_hybrid)
        elif len(orbitals) == 2 and 'p' in orbitals:
            result = G16NBOQMData._hybridization_calculator_sp(nbo_hybrid)
        elif len(orbitals) == 1 and 's' in orbitals:
            result = G16NBOQMData._hybridization_calculator_s()
        else:
            raise ValueError(f'Not implemented for {nbo_hybrid}')

        return result

    @staticmethod
    def _hybridization_calculator_s():

        population = (('s', 1.00), ('p', 0.00), ('d', 0.00))
        pct = (('s%', 100.00), ('p%', 0.00), ('d%', 0.00))
        result = (population, pct)

        return result

    @staticmethod
    def _hybridization_calculator_p():

        population = (('s', 0.00), ('p', 1.00), ('d', 0.00))
        pct = (('s%', 0.00), ('p%', 100.00), ('d%', 0.00))
        result = (population, pct)

        return result

    @staticmethod
    def _hybridization_calculator_d():

        population = (('s', 0.00), ('p', 0.00), ('d', 1.00))
        pct = (('s%', 0.00), ('p%', 0.00), ('d%', 100.00))
        result = (population, pct)

        return result

    @staticmethod
    def _hybridization_calculator_sp(nbo_hybrid):

        numerals = [float(x) for x in re.findall(
            r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", nbo_hybrid)]

        if len(numerals) == 3:
            s_pct, p_pop, p_pct = numerals
        elif len(numerals) == 5:
            s_pct, p_pop, p_pct, d_pop, d_pct = numerals
        else:
            raise ValueError(f'not implemented for {nbo_hybrid}')

        if p_pop < 0.01 or p_pct < 1.00 or s_pct > 99.00:
            result = G16NBOQMData._hybridization_calculator_s()
            return result

        if s_pct < 1.00:
            result = G16NBOQMData._hybridization_calculator_p()
            return result

        s_pop = 1.00
        tot_pop = s_pop + p_pop
        s_pct_norm = (s_pop / tot_pop) * 100
        p_pct_norm = (p_pop / tot_pop) * 100

        population = (('s', 1.00), ('p', round(p_pop, 2)), ('d', 0.00))
        pct = (('s%', round(s_pct_norm, 2)),
               ('p%', round(p_pct_norm, 2)), ('d%', 0.00))
        result = (population, pct)

        return result

    @staticmethod
    def _hybridization_calculator_sd(nbo_hybrid):

        numerals = [float(x) for x in re.findall(
            r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", nbo_hybrid)]

        s_pct, p_pop, p_pct, d_pop, d_pct = numerals

        if d_pop < 0.01 or d_pct < 1.00 or s_pct > 99.00:
            result = G16NBOQMData._hybridization_calculator_s()
            return result

        if s_pct < 1.00:
            result = G16NBOQMData._hybridization_calculator_d()
            return result

        s_pop = 1.00
        tot_pop = s_pop + d_pop
        s_pct_norm = (s_pop / tot_pop) * 100
        d_pct_norm = (d_pop / tot_pop) * 100

        population = (('s', 1.00), ('p', 0.00), ('d', round(d_pop, 2)))
        pct = (('s%', round(s_pct_norm, 2)),
               ('p%', 0.00), ('d%', round(d_pct_norm, 2)))
        result = (population, pct)

        return result

    @staticmethod
    def _hybridization_calculator_pd(nbo_hybrid):

        numerals = [float(x) for x in re.findall(
            r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", nbo_hybrid)]

        s_pct, p_pop, p_pct, d_pop, d_pct = numerals

        if d_pop < 0.01 or d_pct < 1.00 or p_pct > 99.00:
            result = G16NBOQMData._hybridization_calculator_p()
            return result

        if p_pop < 0.01 or p_pct < 1.00 or d_pct > 99.00:
            result = G16NBOQMData._hybridization_calculator_d()
            return result

        tot_pop = p_pop + d_pop
        p_pct_norm = (p_pop / tot_pop) * 100
        d_pct_norm = (d_pop / tot_pop) * 100

        population = (('s', 0.00), ('p', round(p_pop, 2)),
                      ('d', round(d_pop, 2)))
        pct = (('s%', 0.00), ('p%', round(p_pct_norm, 2)),
               ('d%', round(d_pct_norm, 2)))
        result = (population, pct)

        return result

    @staticmethod
    def _hybridization_calculator_spd(nbo_hybrid):

        numerals = [float(x) for x in re.findall(
            r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", nbo_hybrid)]

        s_pct, p_pop, p_pct, d_pop, d_pct = numerals

        if d_pop < 0.01 or d_pct < 1.00:
            result = G16NBOQMData._hybridization_calculator_sp(nbo_hybrid)
            return result

        if p_pop < 0.01 or p_pct < 1.00:
            result = G16NBOQMData._hybridization_calculator_sd(nbo_hybrid)
            return result

        if s_pct < 1.00:
            result = G16NBOQMData._hybridization_calculator_pd(nbo_hybrid)
            return result

        population = (('s', 1.00), ('p', round(p_pop, 2)),
                      ('d', round(d_pop, 2)))
        pct = (('s%', round(s_pct, 2)),
               ('p%', round(p_pct, 2)), ('d%', round(d_pct, 2)))
        result = (population, pct)

        return result

    @staticmethod
    def _GetNBOLewisFullSet(txt, atom_dict):

        for i, line in enumerate(txt):
            if line.find('Bond orbital / Coefficients / Hybrids') > -1:
                txt = txt[i + 2:]
                break

        for i, line in enumerate(txt):
            if line.find('-- non-Lewis --') > -1:
                txt = txt[:i]
                break

        nbo_cr_result = list()
        nbo_lp_result = list()
        nbo_bd_result = list()

        for i, line in enumerate(txt):
            if not any(['CR' in line, 'BD' in line, 'LP' in line]):
                continue

            if 'CR' in line:
                header = line.split()
                info = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
                nbo_idx = int(info[0])
                occupancy = float(info[1])
                nbo_type = 'CR'
                degeneracy = int(info[2])
                atom_idx = int(info[3])
                atom = atom_dict[atom_idx]

                if not atom == re.findall("[a-zA-Z]+", line.split('CR')[-1].split(')')[1])[0]:
                    raise ValueError('atom not match')

                if not atom_idx == int(re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line.split('CR')[-1].split(')')[1])[0]):
                    raise ValueError('atom index not match')

                hybrid_idx = ['s' in x for x in header].index(True)
                hybrid = ''.join(header[hybrid_idx:])
                processed_hybrid = G16NBOQMData._hybridization_calculator(
                    hybrid)
                nbo_cr_result.append(tuple(
                    [nbo_idx, nbo_type, degeneracy, atom, atom_idx, occupancy, hybrid, processed_hybrid]))

            elif 'LP' in line:
                header = line.split()
                info = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
                nbo_idx = int(info[0])
                occupancy = float(info[1])
                nbo_type = 'LP'
                degeneracy = int(info[2])
                atom_idx = int(info[3])
                atom = atom_dict[atom_idx]

                if not atom == re.findall("[a-zA-Z]+", line.split('LP')[-1].split(')')[1])[0]:
                    raise ValueError('atom not match')

                if not atom_idx == int(re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line.split('LP')[-1].split(')')[1])[0]):
                    raise ValueError('atom index not match')

                hybrid_idx = ['s' in x for x in header].index(True)
                hybrid = ''.join(header[hybrid_idx:])
                processed_hybrid = G16NBOQMData._hybridization_calculator(
                    hybrid)
                nbo_lp_result.append(tuple(
                    [nbo_idx, nbo_type, degeneracy, atom, atom_idx, occupancy, hybrid, processed_hybrid]))

            elif 'BD' in line:
                header = line.split()
                info = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
                nbo_idx = int(info[0])
                occupancy = float(info[1])
                nbo_type = 'BD'
                degeneracy = int(info[2])
                atom1_idx = int(info[3])
                atom1 = atom_dict[atom1_idx]
                atom2_idx = int(info[4])
                atom2 = atom_dict[atom2_idx]

                if not atom1 == re.findall("[a-zA-Z]+", line.split(')')[-1].split('-')[0])[0]:
                    raise ValueError('atom not match')

                if not atom1_idx == int(re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line.split(')')[-1].split('-')[0])[0]):
                    raise ValueError('atom index not match')

                if not atom2 == re.findall("[a-zA-Z]+", line.split(')')[-1].split('-')[1])[0]:
                    raise ValueError('atom not match')

                if not atom2_idx == int(re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line.split(')')[-1].split('-')[1])[0]):
                    raise ValueError('atom index not match')

                line2s = txt[i+1]
                line2 = txt[i+1].split()

                if not line2[0] == '(':
                    raise ValueError('log format not expected, check input')

                big_str = ''.join(line2)
                numerals = re.findall(
                    r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", big_str)
                c_a = float(numerals[1])
                c_a_square_pct = float(numerals[0])

                if abs(c_a*c_a - (c_a_square_pct/100)) > 0.0005:
                    raise ValueError(
                        'incosistent natural polarization coefficients')

                if not atom1 == re.findall("[a-zA-Z]+", line2s.split('*')[-1])[0]:
                    raise ValueError('incosistent atomic label')

                if not atom1_idx == int(re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line2s.split('*')[-1])[0]):
                    raise ValueError('incosistent atomic order')

                hybrid1_idx = ['s' in x for x in line2].index(True)
                hybrid1 = ''.join(line2[hybrid1_idx:])
                if not 's' == hybrid1[0]:
                    raise ValueError('elec config parsing error')

                j = i+2
                k = 0
                while True:
                    k += 1
                    l = txt[j].split()
                    if not l[0] == '(':
                        j += 1
                    else:
                        line3 = l
                        line3s = txt[j]
                        break

                    if k == 30:
                        raise ValueError(
                            'log format not expected, check input')

                big_str_2 = ''.join(line3)

                numerals_2 = re.findall(
                    r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", big_str_2)
                c_b = float(numerals_2[1])
                c_b_square_pct = float(numerals_2[0])

                if abs(c_b*c_b - (c_b_square_pct/100)) > 0.0005:
                    raise ValueError(
                        'incosistent natural polarization coefficients')

                if not atom2 == re.findall("[a-zA-Z]+", line3s.split('*')[-1])[0]:
                    raise ValueError('incosistent atomic label')

                if not atom2_idx == int(re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line3s.split('*')[-1])[0]):
                    raise ValueError('incosistent atomic order')

                hybrid2_idx = ['s' in x for x in line3].index(True)
                hybrid2 = ''.join(line3[hybrid2_idx:])
                if not 's' == hybrid2[0]:
                    raise ValueError('elec config parsing error')

                natural_ionicity = abs(
                    (c_a*c_a - c_b*c_b) / (c_a*c_a + c_b*c_b))

                processed_hybrid1 = G16NBOQMData._hybridization_calculator(
                    hybrid1)
                processed_hybrid2 = G16NBOQMData._hybridization_calculator(
                    hybrid2)

                bond_char_atom1 = [c_a * i for i in [x[1]
                                                     for x in processed_hybrid1[0]]]
                bond_char_atom2 = [c_b * i for i in [x[1]
                                                     for x in processed_hybrid2[0]]]

                bond_s, bond_p, bond_d = [sum(v) for v in zip(
                    bond_char_atom1, bond_char_atom2)]
                bond_tot = bond_s + bond_p + bond_d
                bond_s_pct = (bond_s / bond_tot) * 100
                bond_p_pct = (bond_p / bond_tot) * 100
                bond_d_pct = (bond_d / bond_tot) * 100

                bd_pop = (('s', round(bond_s, 2)),
                          ('p', round(bond_p, 2)), ('d', round(bond_d, 2)))
                bd_pct = (('s%', round(bond_s_pct, 2)), ('p%', round(
                    bond_p_pct, 2)), ('d%', round(bond_d_pct, 2)))
                bond_char = (bd_pop, bd_pct)

                hybrid1_result = tuple([atom1, atom1_idx, c_a, round(
                    c_a*c_a, 4), hybrid1, processed_hybrid1])
                hybrid2_result = tuple([atom2, atom2_idx, c_b, round(
                    c_b*c_b, 4), hybrid2, processed_hybrid2])
                bd_result = tuple([nbo_idx, nbo_type, degeneracy, atom1, atom1_idx,
                                  atom2, atom2_idx, occupancy, round(natural_ionicity, 4), bond_char])

                nbo_bd_result.append(
                    tuple([bd_result, hybrid1_result, hybrid2_result]))

            result = tuple([tuple(nbo_cr_result), tuple(
                nbo_lp_result), tuple(nbo_bd_result)])

        return result

    def GetNBOLewisFullSet(self):
        if self.open_shell:
            self.nbo_lewis_full_alpha_spin_orbital = self._GetNBOLewisFullSet(
                self.txt_alpha, self.atom_dict)
            self.nbo_lewis_full_beta_spin_orbital = self._GetNBOLewisFullSet(
                self.txt_beta, self.atom_dict)
            self.nbo_lewis_full_closed_shell_overall = None
        else:
            self.nbo_lewis_full_alpha_spin_orbital = None
            self.nbo_lewis_full_beta_spin_orbital = None
            self.nbo_lewis_full_closed_shell_overall = self._GetNBOLewisFullSet(
                self.log, self.atom_dict)

        if self.open_shell:
            self.nbo_lewis_full_alpha_spin_orbital_cr, self.nbo_lewis_full_alpha_spin_orbital_lp, self.nbo_lewis_full_alpha_spin_orbital_bd = self.nbo_lewis_full_alpha_spin_orbital
            self.nbo_lewis_full_beta_spin_orbital_cr, self.nbo_lewis_full_beta_spin_orbital_lp, self.nbo_lewis_full_beta_spin_orbital_bd = self.nbo_lewis_full_beta_spin_orbital
            self.nbo_lewis_full_closed_shell_overall_cr, self.nbo_lewis_full_closed_shell_overall_lp, self.nbo_lewis_full_closed_shell_overall_bd = (
                None, None, None)
        else:
            self.nbo_lewis_full_alpha_spin_orbital_cr, self.nbo_lewis_full_alpha_spin_orbital_lp, self.nbo_lewis_full_alpha_spin_orbital_bd = (
                None, None, None)
            self.nbo_lewis_full_beta_spin_orbital_cr, self.nbo_lewis_full_beta_spin_orbital_lp, self.nbo_lewis_full_beta_spin_orbital_bd = (
                None, None, None)
            self.nbo_lewis_full_closed_shell_overall_cr, self.nbo_lewis_full_closed_shell_overall_lp, self.nbo_lewis_full_closed_shell_overall_bd = self.nbo_lewis_full_closed_shell_overall

        self.nbo_lewis_full_meta = tuple(['CR', 'LP', 'BD'])
        self.nbo_lewis_full_cr_meta = tuple(
            ['nbo_idx', 'nbo_type', 'degeneracy', 'atom', 'atom_idx', 'occupancy', 'hybrid', 'processed_hybrid'])
        self.nbo_lewis_full_lp_meta = tuple(
            ['nbo_idx', 'nbo_type', 'degeneracy', 'atom', 'atom_idx', 'occupancy', 'hybrid', 'processed_hybrid'])
        self.nbo_lewis_full_bd_meta = tuple(
            ['bond_result', 'hybrid1_result', 'hybrid2_result'])

        self.nbo_lewis_full_bd_bond_meta = tuple(['nbo_idx', 'nbo_type', 'degeneracy', 'atom1',
                                                 'atom1_idx', 'atom2', 'atom2_idx', 'occupancy', 'natural_ionicity', 'calculated_bond_char'])
        self.nbo_lewis_full_bd_hybrid1_meta = tuple(
            ['atom1', 'atom1_idx', 'ca', 'ca*ca', 'hybrid1', 'processed_hybrid1'])
        self.nbo_lewis_full_bd_hybrid2_meta = tuple(
            ['atom2', 'atom2_idx', 'cb', 'cb*cb', 'hybrid2', 'processed_hybrid2'])

    @staticmethod
    def _ParseNBOLewisEnergy(txt, atom_dict):

        nbo_cr_result = list()
        nbo_lp_result = list()
        nbo_bd_result = list()

        for i, line in enumerate(txt):
            if not any(['CR' in line, 'BD' in line, 'LP' in line]):
                continue

            if 'CR' in line:
                header = line.split()
                info = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
                nbo_idx = int(info[0])
                degeneracy = int(info[1])
                atom_idx = int(info[2])
                occupancy = float(info[3])
                energy = float(info[4])

                nbo_type = 'CR'

                atom = atom_dict[atom_idx]

                if not atom == re.findall("[a-zA-Z]+", line.split('CR')[-1].split(')')[1])[0]:
                    raise ValueError('atom not match')

                if not atom_idx == int(re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line.split('CR')[-1].split(')')[1])[0]):
                    raise ValueError('atom index not match')

                nbo_cr_result.append(
                    tuple([nbo_idx, nbo_type, degeneracy, atom, atom_idx, occupancy, energy]))

            if 'LP' in line:
                header = line.split()
                info = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
                nbo_idx = int(info[0])
                degeneracy = int(info[1])
                atom_idx = int(info[2])
                occupancy = float(info[3])
                energy = float(info[4])

                nbo_type = 'LP'

                atom = atom_dict[atom_idx]

                if not atom == re.findall("[a-zA-Z]+", line.split('LP')[-1].split(')')[1])[0]:
                    raise ValueError('atom not match')

                if not atom_idx == int(re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line.split('LP')[-1].split(')')[1])[0]):
                    raise ValueError('atom index not match')

                nbo_lp_result.append(
                    tuple([nbo_idx, nbo_type, degeneracy, atom, atom_idx, occupancy, energy]))

            if 'BD' in line:
                header = line.split()
                info = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
                nbo_idx = int(info[0])
                degeneracy = int(info[1])
                atom1_idx = int(info[2])
                atom2_idx = int(info[3])
                occupancy = float(info[4])
                energy = float(info[5])

                nbo_type = 'BD'

                atom1 = atom_dict[atom1_idx]
                atom2 = atom_dict[atom2_idx]

                if not atom1 == re.findall("[a-zA-Z]+", line.split(')')[1].split('-')[0])[0]:
                    raise ValueError('atom not match')

                if not atom1_idx == int(re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line.split(')')[1].split('-')[0])[0]):
                    raise ValueError('atom index not match')

                if not atom2 == re.findall("[a-zA-Z]+", line.split(')')[1].split('-')[1])[0]:
                    raise ValueError('atom not match')

                if not atom2_idx == int(re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line.split(')')[1].split('-')[1])[0]):
                    raise ValueError('atom index not match')

                nbo_bd_result.append(tuple(
                    [nbo_idx, nbo_type, degeneracy, atom1, atom1_idx, atom2, atom2_idx, occupancy, energy]))

        return nbo_cr_result, nbo_lp_result, nbo_bd_result

    @staticmethod
    def _GetNBOLewisEnergy(txt, atom_dict):
        for i, line in enumerate(txt):
            if line.find('NATURAL BOND ORBITALS (Summary):') > -1:
                txt = txt[i + 2:]
                break

        for i, line in enumerate(txt):
            if line.find('-- Lewis --') > -1:
                start_idx1 = i
                break

        for i, line in enumerate(txt):
            if line.find('-- non-Lewis --') > -1:
                end_idx1 = i
                break

        txt1 = txt[start_idx1+1:end_idx1]

        seg = txt[end_idx1+1:]
        start_idx2 = None
        end_idx2 = None

        for i, line in enumerate(seg):
            if line.find('-- Lewis --') > -1:
                start_idx2 = i
                break

        for i, line in enumerate(seg):
            if line.find('-- non-Lewis --') > -1:
                end_idx2 = i
                break

        txt2, txt3, txt4, txt5 = None, None, None, None

        if start_idx2 and end_idx2:
            txt2 = seg[start_idx2+1:end_idx2]

            seg2 = seg[end_idx2+1:]
            start_idx3 = None
            end_idx3 = None

            for i, line in enumerate(seg2):
                if line.find('-- Lewis --') > -1:
                    start_idx3 = i
                    break

            for i, line in enumerate(seg2):
                if line.find('-- non-Lewis --') > -1:
                    end_idx3 = i
                    break

            if start_idx3 and end_idx3:
                txt3 = seg2[start_idx3+1:end_idx3]

                seg3 = seg2[end_idx3+1:]
                start_idx4 = None
                end_idx4 = None

                for i, line in enumerate(seg3):
                    if line.find('-- Lewis --') > -1:
                        start_idx4 = i
                        break

                for i, line in enumerate(seg3):
                    if line.find('-- non-Lewis --') > -1:
                        end_idx4 = i
                        break

                if start_idx4 and end_idx4:
                    txt4 = seg3[start_idx4+1:end_idx4]

                    seg4 = seg3[end_idx4+1:]
                    start_idx5 = None
                    end_idx5 = None

                    for i, line in enumerate(seg4):
                        if line.find('-- Lewis --') > -1:
                            start_idx5 = i
                            break

                    for i, line in enumerate(seg4):
                        if line.find('-- non-Lewis --') > -1:
                            end_idx5 = i
                            break

                    if start_idx5 and end_idx5:
                        txt5 = seg4[start_idx5+1:end_idx5]

                        seg5 = seg4[end_idx5+1:]
                        start_idx6 = None
                        end_idx6 = None

                        for i, line in enumerate(seg5):
                            if line.find('-- Lewis --') > -1:
                                start_idx6 = i
                                break

                        for i, line in enumerate(seg5):
                            if line.find('-- non-Lewis --') > -1:
                                end_idx6 = i
                                break

                        if start_idx6 or start_idx6:
                            raise ValueError('unexpected log format')

        nbo_cr_result = list()
        nbo_lp_result = list()
        nbo_bd_result = list()

        nbo_cr_result.extend(
            G16NBOQMData._ParseNBOLewisEnergy(txt1, atom_dict)[0])
        nbo_lp_result.extend(
            G16NBOQMData._ParseNBOLewisEnergy(txt1, atom_dict)[1])
        nbo_bd_result.extend(
            G16NBOQMData._ParseNBOLewisEnergy(txt1, atom_dict)[2])

        if txt2:
            nbo_cr_result.extend(
                G16NBOQMData._ParseNBOLewisEnergy(txt2, atom_dict)[0])
            nbo_lp_result.extend(
                G16NBOQMData._ParseNBOLewisEnergy(txt2, atom_dict)[1])
            nbo_bd_result.extend(
                G16NBOQMData._ParseNBOLewisEnergy(txt2, atom_dict)[2])

        if txt3:
            nbo_cr_result.extend(
                G16NBOQMData._ParseNBOLewisEnergy(txt3, atom_dict)[0])
            nbo_lp_result.extend(
                G16NBOQMData._ParseNBOLewisEnergy(txt3, atom_dict)[1])
            nbo_bd_result.extend(
                G16NBOQMData._ParseNBOLewisEnergy(txt3, atom_dict)[2])

        if txt4:
            nbo_cr_result.extend(
                G16NBOQMData._ParseNBOLewisEnergy(txt4, atom_dict)[0])
            nbo_lp_result.extend(
                G16NBOQMData._ParseNBOLewisEnergy(txt4, atom_dict)[1])
            nbo_bd_result.extend(
                G16NBOQMData._ParseNBOLewisEnergy(txt4, atom_dict)[2])

        if txt5:
            nbo_cr_result.extend(
                G16NBOQMData._ParseNBOLewisEnergy(txt5, atom_dict)[0])
            nbo_lp_result.extend(
                G16NBOQMData._ParseNBOLewisEnergy(txt5, atom_dict)[1])
            nbo_bd_result.extend(
                G16NBOQMData._ParseNBOLewisEnergy(txt5, atom_dict)[2])

        result = [tuple(nbo_cr_result), tuple(
            nbo_lp_result), tuple(nbo_bd_result)]

        return tuple(result)

    def GetNBOLewisEnergy(self):
        if self.open_shell:
            self.nbo_lewis_energy_alpha_spin_orbital = self._GetNBOLewisEnergy(
                self.txt_alpha, self.atom_dict)
            self.nbo_lewis_energy_beta_spin_orbital = self._GetNBOLewisEnergy(
                self.txt_beta, self.atom_dict)
            self.nbo_lewis_energy_closed_shell_overall = None
        else:
            self.nbo_lewis_energy_alpha_spin_orbital = None
            self.nbo_lewis_energy_beta_spin_orbital = None
            self.nbo_lewis_energy_closed_shell_overall = self._GetNBOLewisEnergy(
                self.log, self.atom_dict)
        self.nbo_lewis_energy_meta = tuple(['CR', 'LP', 'BD'])
        self.nbo_lewis_energy_cr_meta = tuple(
            ['nbo_idx', 'nbo_type', 'degeneracy', 'atom', 'atom_idx', 'occupancy', 'energy'])
        self.nbo_lewis_energy_lp_meta = tuple(
            ['nbo_idx', 'nbo_type', 'degeneracy', 'atom', 'atom_idx', 'occupancy', 'energy'])
        self.nbo_lewis_energy_bd_meta = tuple(
            ['nbo_idx', 'nbo_type', 'degeneracy', 'atom1', 'atom1_idx', 'atom2', 'atom2_idx', 'occupancy', 'energy'])

    def GetNLMOBondOrderMatrix(self):

        if not self.open_shell:

            txt = self.log

            for i, line in enumerate(txt):
                if line.find('Atom-Atom Net Linear NLMO/NPA Bond Orders:') > -1:
                    txt = txt[i + 4:]
                    break

            for i, line in enumerate(txt):
                if line.find('Linear NLMO/NPA Bond') > -1:
                    txt = txt[:i]
                    break

            self.nlmo = G16NBOQMData._parseLargeMatrix(txt)
        else:
            self.nlmo = None

    def GetNLMOBondOrderMatrixbyAtom(self):
        txt = self.log

        if not self.open_shell:
            bond_index_matrix = list()

            for i, line in enumerate(txt):
                if line.find('Linear NLMO/NPA Bond Orders, Totals by Atom:') > -1:
                    txt = txt[i + 4:]
                    break

            for i, line in enumerate(txt):
                if not line:
                    break

                bond_index_matrix.append(float(line.split()[2]))
            self.nlmo_atom = tuple(bond_index_matrix)
        else:
            self.nlmo_atom = None

    @staticmethod
    def _parseNMRMatrix(txt):
        # width = atom order
        # height = nbo

        data_l = dict()
        data_nl = dict()

        for i, line in enumerate(txt):
            if not line:
                continue

            try:
                _idx = line.split()[0][:-1]
            except IndexError:
                continue

            if not _idx.isnumeric():
                del _idx
                continue
            else:
                atom_idx = int(_idx)
                if atom_idx not in data_l.keys():
                    data_l[atom_idx] = list()
                    data_nl[atom_idx] = list()

                if 'L' in line:
                    start = line.split().index('L')
                    data_l[atom_idx].extend([float(x)
                                            for x in line.split()[start+1:]])
                    data_nl[atom_idx].extend([float(x)
                                             for x in txt[i+1].split()[1:]])

        width1 = len(data_l[1])
        height1 = len(data_l.keys())

        width2 = len(data_nl[1])
        height2 = len(data_nl.keys())

        if not height1 == height1:
            raise ValueError('inconsistent matrix dimension, parser error')

        if not width1 == width2:
            raise ValueError('inconsistent matrix dimension, parser error')

        result_l = list()
        result_nl = list()
        for i in range(height1):
            atom_idx = i + 1
            result_l.append(tuple(data_l[atom_idx]))
            result_nl.append(tuple(data_nl[atom_idx]))

        return tuple(result_l), tuple(result_nl)

    def GetNBONMR(self):
        if not self.open_shell:

            txt = self.log

            for i, line in enumerate(txt):
                if line.find('Summary of isotropic NMR chemical shielding') > -1:
                    txt = txt[i + 5:]
                    break

            for i, line in enumerate(txt):
                if line.find('NBO analysis completed') > -1:
                    txt = txt[:i]
                    break

            self.nmr_lewis_matrix, self.nmr_non_lewis_matrix = G16NBOQMData._parseNMRMatrix(
                txt)
            self.nmr_meta = tuple(['width: atom order', 'height: nbo'])
        else:
            self.nmr_lewis_matrix, self.nmr_non_lewis_matrix = (None, None)
            self.nmr_meta = None

    def SCFOrbitalEnergy(self):

        txt = self.log
        energy_levels = list()

        for i, line in enumerate(txt):
            if line.find('Population analysis using the SCF Density.') > -1:
                txt = txt[i:]
                break

        for i, line in enumerate(txt):
            if line.find('Molecular Orbital') > -1:
                txt = txt[:i]
                break

        for i, line in enumerate(txt):
            if line.find('The electronic state is') > -1:
                txt = txt[i+1:]
                break

        for i, line in enumerate(txt):
            if 'eigenvalues' in line:
                level = re.findall(
                    r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
                energy_levels.extend(tuple([float(x) for x in level]))

        levels = np.array(energy_levels)
        homo = float(levels[np.where(levels < 0, levels, -np.inf).argmax()])
        lumo = float(levels[np.where(levels > 0, levels, np.inf).argmin()])

        self.scf_levels = tuple([float(x) for x in energy_levels])
        self.homo = homo
        self.lumo = lumo

    def GetCSPA(self):
        if not self.open_shell:
            data = self.glog._cclib_results
            m = CSPA(data)
            m.calculate()
            self.CSPA = tuple([tuple(frag) for frag in m.fragresults[0]])
        else:
            self.CSPA = None

    def GetDataDict(self):
        self.data = {'CPU': self.CPU,
                     'atom_dict': self.atom_dict,
                     'basename': self.basename,
                     'bond_length_matrix': self.bond_length_matrix,
                     'charge': self.charge,
                     'data_type': self.data_type,
                     'elec_config_alpha_spin_orbital': self.elec_config_alpha_spin_orbital,
                     'elec_config_beta_spin_orbital': self.elec_config_beta_spin_orbital,
                     'elec_config_closed_shell_overall': self.elec_config_closed_shell_overall,
                     'elec_config_open_shell_overall': self.elec_config_open_shell_overall,
                     'energy': self.energy,
                     'file': self.file,
                     'hirshfeld_charges': self.hirshfeld_charges,
                     'hirshfeld_cm5_charges': self.hirshfeld_cm5_charges,
                     'hirshfeld_dipoles_x': self.hirshfeld_dipoles_x,
                     'hirshfeld_dipoles_y': self.hirshfeld_dipoles_y,
                     'hirshfeld_dipoles_z': self.hirshfeld_dipoles_z,
                     'hirshfeld_spin_density': self.hirshfeld_spin_density,
                     'homo': self.homo,
                     'id': self.id,
                     'lumo': self.lumo,
                     'method': self.method,
                     'mulliken_charge': self.mulliken_charge,
                     'mulliken_dipole_tot': self.mulliken_dipole_tot,
                     'mulliken_dipole_x': self.mulliken_dipole_x,
                     'mulliken_dipole_xx': self.mulliken_dipole_xx,
                     'mulliken_dipole_xy': self.mulliken_dipole_xy,
                     'mulliken_dipole_xz': self.mulliken_dipole_xz,
                     'mulliken_dipole_y': self.mulliken_dipole_y,
                     'mulliken_dipole_yy': self.mulliken_dipole_yy,
                     'mulliken_dipole_yz': self.mulliken_dipole_yz,
                     'mulliken_dipole_z': self.mulliken_dipole_z,
                     'mulliken_dipole_zz': self.mulliken_dipole_zz,
                     'mulliken_quadrupoles': self.mulliken_quadrupoles,
                     'mulliken_condensed_charge_matrix': self.mulliken_condensed_charge_matrix,
                     'mulliken_spin': self.mulliken_spin,
                     'multiplicity': self.multiplicity,
                     'nao_pop_alpha_spin_orbital': self.nao_pop_alpha_spin_orbital,
                     'nao_pop_beta_spin_orbital': self.nao_pop_beta_spin_orbital,
                     'nao_pop_closed_shell_overall': self.nao_pop_closed_shell_overall,
                     'nao_pop_meta': self.nao_pop_meta,
                     'nao_pop_open_shell_overall': self.nao_pop_open_shell_overall,
                     '1s_val_occ': self._1s_val,
                     '2s_val_occ': self._2s_val,
                     '2p_val_occ': self._2p_val,
                     '3s_val_occ': self._3s_val,
                     '3p_val_occ': self._3p_val,
                     '4s_val_occ': self._4s_val,
                     '4p_val_occ': self._4p_val,
                     'nbi_alpha_spin_orbital': self.nbi_alpha_spin_orbital,
                     'nbi_beta_spin_orbital': self.nbi_beta_spin_orbital,
                     'nbi_closed_shell_overall': self.nbi_closed_shell_overall,
                     'nbi_open_shell_overall': self.nbi_open_shell_overall,
                     'nbo_lewis_alpha_spin_orbital': self.nbo_lewis_alpha_spin_orbital,
                     'nbo_lewis_beta_spin_orbital': self.nbo_lewis_beta_spin_orbital,
                     'nbo_lewis_closed_shell_overall': self.nbo_lewis_closed_shell_overall,
                     'nbo_lewis_energy_alpha_spin_orbital': self.nbo_lewis_energy_alpha_spin_orbital,
                     'nbo_lewis_energy_bd_meta': self.nbo_lewis_energy_bd_meta,
                     'nbo_lewis_energy_beta_spin_orbital': self.nbo_lewis_energy_beta_spin_orbital,
                     'nbo_lewis_energy_closed_shell_overall': self.nbo_lewis_energy_closed_shell_overall,
                     'nbo_lewis_energy_cr_meta': self.nbo_lewis_energy_cr_meta,
                     'nbo_lewis_energy_lp_meta': self.nbo_lewis_energy_lp_meta,
                     'nbo_lewis_energy_meta': self.nbo_lewis_energy_meta,
                     'nbo_lewis_full_alpha_spin_orbital_bd': self.nbo_lewis_full_alpha_spin_orbital_bd,
                     'nbo_lewis_full_alpha_spin_orbital_cr': self.nbo_lewis_full_alpha_spin_orbital_cr,
                     'nbo_lewis_full_alpha_spin_orbital_lp': self.nbo_lewis_full_alpha_spin_orbital_lp,
                     'nbo_lewis_full_bd_bond_meta': self.nbo_lewis_full_bd_bond_meta,
                     'nbo_lewis_full_bd_hybrid1_meta': self.nbo_lewis_full_bd_hybrid1_meta,
                     'nbo_lewis_full_bd_hybrid2_meta': self.nbo_lewis_full_bd_hybrid2_meta,
                     'nbo_lewis_full_bd_meta': self.nbo_lewis_full_bd_meta,
                     'nbo_lewis_full_beta_spin_orbital_bd': self.nbo_lewis_full_beta_spin_orbital_bd,
                     'nbo_lewis_full_beta_spin_orbital_cr': self.nbo_lewis_full_beta_spin_orbital_cr,
                     'nbo_lewis_full_beta_spin_orbital_lp': self.nbo_lewis_full_beta_spin_orbital_lp,
                     'nbo_lewis_full_closed_shell_overall_bd': self.nbo_lewis_full_closed_shell_overall_bd,
                     'nbo_lewis_full_closed_shell_overall_cr': self.nbo_lewis_full_closed_shell_overall_cr,
                     'nbo_lewis_full_closed_shell_overall_lp': self.nbo_lewis_full_closed_shell_overall_lp,
                     'nbo_lewis_full_cr_meta': self.nbo_lewis_full_cr_meta,
                     'nbo_lewis_full_lp_meta': self.nbo_lewis_full_lp_meta,
                     'nbo_lewis_full_meta': self.nbo_lewis_full_meta,
                     'nbo_lewis_meta': self.nbo_lewis_meta,
                     'nbo_lewis_open_shell_overall': self.nbo_lewis_open_shell_overall,
                     'nlmo': self.nlmo,
                     'nlmo_atom': self.nlmo_atom,
                     'nmr_lewis_matrix': self.nmr_lewis_matrix,
                     'nmr_meta': self.nmr_meta,
                     'nmr_non_lewis_matrix': self.nmr_non_lewis_matrix,
                     'npa_charge_alpha_spin_orbital': self.npa_charge_alpha_spin_orbital,
                     'npa_charge_beta_spin_orbital': self.npa_charge_beta_spin_orbital,
                     'npa_charge_closed_shell_overall': self.npa_charge_closed_shell_overall,
                     'npa_charge_meta': self.npa_charge_meta,
                     'npa_charge_open_shell_overall': self.npa_charge_open_shell_overall,
                     'npa_charge': self.npa_charge,
                     'npa_spin_density': self.npa_spin_density,
                     'npa_wiberg_bdx_alpha_spin_orbital': self.npa_wiberg_bdx_alpha_spin_orbital,
                     'npa_wiberg_bdx_beta_spin_orbital': self.npa_wiberg_bdx_beta_spin_orbital,
                     'npa_wiberg_bdx_by_atom_alpha_spin_orbital': self.npa_wiberg_bdx_by_atom_alpha_spin_orbital,
                     'npa_wiberg_bdx_by_atom_beta_spin_orbital': self.npa_wiberg_bdx_by_atom_beta_spin_orbital,
                     'npa_wiberg_bdx_by_atom_closed_shell_overall': self.npa_wiberg_bdx_by_atom_closed_shell_overall,
                     'npa_wiberg_bdx_by_atom_open_shell_overall': self.npa_wiberg_bdx_by_atom_open_shell_overall,
                     'npa_wiberg_bdx_closed_shell_overall': self.npa_wiberg_bdx_closed_shell_overall,
                     'npa_wiberg_bdx_open_shell_overall': self.npa_wiberg_bdx_open_shell_overall,
                     'open_shell': self.open_shell,
                     'scf_levels': self.scf_levels,
                     'CSPA': self.CSPA,
                     'smi': self.smi,
                     'success': self.success,
                     'xyz': self.xyz}


class XtbLog:
    def __init__(self, file):
        # default values for thermochemical calculations
        if '.log' not in file:
            raise TypeError('A xtb .log file must be provided')

        self.file = file
        self.name = os.path.basename(file)

        self.GetTermination()
        if not self.termination:
            pass
            # self.GetError()
        else:
            self.GetFreq()
            self.GetE()

    def GetTermination(self):
        with open(self.file) as fh:
            for line in (fh):
                if line.find("normal termination") > -1:
                    self.termination = True
                    return True
            self.termination = False

    def GetFreq(self):
        with open(self.file) as fh:
            txt = fh.readlines()

        txt = [x.strip() for x in txt]
        for i, line in enumerate(txt):
            if line.find('Frequency Printout') > -1:
                txt = txt[i + 3:]
                break

        waveNums = []
        for i, line in enumerate(txt):
            if line.find('reduced masses') > -1:
                txt = txt[i + 1:]
                break
            m = re.findall('\s+(-?\d+\.\d+)', line)
            if m:
                for match in m:
                    waveNums.append(float(match.strip()))

        for i, line in enumerate(txt):
            if line.find('IR intensities') > -1:
                txt = txt[i + 1:]
                break

        intensities = []
        for i, line in enumerate(txt):
            if line.find('Raman intensities') > -1:
                txt = txt[i + 1:]
                break
            m = re.findall('\d+:\s+(\d+\.\d+)', line)
            if m:
                for match in m:
                    intensities.append(float(match))

        waveNums, intensities = list(
            zip(*[(w, i) for w, i in zip(waveNums, intensities) if w != 0]))

        if waveNums and intensities and len(waveNums) == len(intensities):
            self.wavenum = waveNums
            self.ir_intensities = intensities

    def GetE(self):
        with open(self.file) as fh:
            txt = fh.readlines()

        txt = [x.strip() for x in txt]
        for i, line in enumerate(txt):
            m = re.search('TOTAL ENERGY\s+(-?\d+\.\d+)', line)
            if m:
                self.E = m[1]
                continue
            m = re.search('TOTAL FREE ENERGY\s+(-?\d+\.\d+)', line)
            if m:
                self.G = float(m[1])
