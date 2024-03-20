import os
import json
import numpy as np
import pandas as pd
from rdmc import RDKitMol
from rdkit import Chem
pd.set_option('mode.chained_assignment', None)

class json2csv():
    def __init__(self, base_file, plus_file=None, minus_file=None, base_only=False):
        """
        :param base_file: Path to json file for base molecule.
        :param plus_file: Path to json file for plus molecule.
        :param minus_file: Path to json file for minus molecule.
        :param base_only: Whether to only use base molecule. (Default: False)
        """
        self.base_only = base_only
        self.base_data = self.load_file(base_file)
        base_keys = self.base_data.keys()
        if not base_only:
            self.plus_data = self.load_file(plus_file)
            self.minus_data = self.load_file(minus_file)
            plus_keys = self.plus_data.keys()
            minus_keys = self.minus_data.keys()
            self.keys = list(set(base_keys) & set(plus_keys) & set(minus_keys))
        else:
            self.keys = list(base_keys)

    def save_descriptors(self, output_path):
        """
        Save result to a .csv file containing all the atom/bond/molecule descriptors
        mentioned in :code:`property_names`.
        """
        df_mols = pd.DataFrame()
        df_constraints = pd.DataFrame()
        smiles_list = []
        for id in self.keys:
            base_data = self.base_data[id]
            plus_data = self.plus_data[id]
            minus_data = self.minus_data[id]
            assert base_data['smi'] == plus_data['smi'] == minus_data['smi']
            if base_data['smi'] in smiles_list:
                continue
            else:
                smiles_list.append(base_data['smi'])
            d_mol = dict()
            d_constraints = dict()
            for key in property_names:
                mol = make_mol(base_data['smi'])
                if key == 'smiles':
                    d_mol[key] = base_data['smi']
                elif key == 'hirshfeld_charges':
                    d_mol[key] = np.array(base_data[key])
                    if round(sum(d_mol[key])) != base_data['charge']:
                        raise ValueError(
                            "The sum of hirshfeld charges for base molecule is not equal to the charge of the molecule!")
                    d_constraints[key] = round(sum(d_mol[key]))
                elif key == 'hirshfeld_charges_plus':
                    d_mol[key] = np.array(plus_data['hirshfeld_charges'])
                    if round(sum(d_mol[key])) != plus_data['charge']:
                        raise ValueError(
                            "The sum of hirshfeld charges for plus molecule is not equal to the charge of the molecule!")
                    d_constraints[key] = round(sum(d_mol[key]))
                elif key == 'hirshfeld_charges_minus':
                    d_mol[key] = np.array(minus_data['hirshfeld_charges'])
                    if round(sum(d_mol[key])) != minus_data['charge']:
                        raise ValueError(
                            "The sum of hirshfeld charges for minus molecule is not equal to the charge of the molecule!")
                    d_constraints[key] = round(sum(d_mol[key]))
                elif key == 'hirshfeld_fukui_neu':
                    v = np.array(plus_data['hirshfeld_charges']) - \
                        np.array(base_data['hirshfeld_charges'])
                    d_mol[key] = v
                    d_constraints[key] = round(sum(v))
                elif key == 'hirshfeld_fukui_elec':
                    v = np.array(base_data['hirshfeld_charges']) - \
                        np.array(minus_data['hirshfeld_charges'])
                    d_mol[key] = v
                    d_constraints[key] = round(sum(v))
                elif key == 'hirshfeld_parr_neu':
                    v = np.array(minus_data['hirshfeld_spin_density'])
                    d_mol[key] = v
                    d_constraints[key] = round(sum(v))
                elif key == 'hirshfeld_parr_elec':
                    v = np.array(plus_data['hirshfeld_spin_density'])
                    d_mol[key] = v
                    d_constraints[key] = round(sum(v))
                elif key == 'npa_parr_neu':
                    v = np.array(minus_data['npa_spin_density'])
                    d_mol[key] = v
                    d_constraints[key] = round(sum(v))
                elif key == 'npa_parr_elec':
                    v = np.array(plus_data['npa_spin_density'])
                    d_mol[key] = v
                    d_constraints[key] = round(sum(v))
                elif key == 'shielding_constants':
                    nmr_lewis_matrix = base_data['nmr_lewis_matrix']
                    nmr_non_lewis_matrix = base_data['nmr_non_lewis_matrix']
                    v = self.get_shielding_constants(
                        nmr_lewis_matrix, nmr_non_lewis_matrix)
                    d_mol[key] = np.array(v)
                elif key == 'bond_index_matrix':
                    v = np.array(
                        base_data['npa_wiberg_bdx_closed_shell_overall'])
                    bond_target_arranged = []
                    for bond in mol.GetBonds():
                        bond_target_arranged.append(
                            v[bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
                    v = bond_target_arranged
                    d_mol[key] = np.array(v)
                elif key == 'natural_ionicity':
                    smi = base_data['smi']
                    nbo_lewis_full_orbital_bd = base_data['nbo_lewis_full_closed_shell_overall_bd']
                    v = self.get_nbo_lewis_orbital_bd(smi, nbo_lewis_full_orbital_bd)[
                        'natural_ionicity']
                    v = np.array(v)
                    bond_target_arranged = []
                    for bond in mol.GetBonds():
                        bond_target_arranged.append(
                            v[bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
                    v = bond_target_arranged
                    d_mol[key] = np.array(v)
                elif key == 'bond_length_matrix':
                    v = np.array(base_data['bond_length_matrix'])
                    bond_target_arranged = []
                    for bond in mol.GetBonds():
                        bond_target_arranged.append(
                            v[bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
                    v = bond_target_arranged
                    d_mol[key] = np.array(v)
                elif key == 'bond_charge':
                    v = np.array(base_data['mulliken_condensed_charge_matrix'])
                    bond_target_arranged = []
                    for bond in mol.GetBonds():
                        bond_target_arranged.append(
                            v[bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
                    v = bond_target_arranged
                    d_mol[key] = np.array(v)
                elif 'charge' in key:
                    if 'plus' in key:
                        d_mol[key] = np.array(plus_data[key[:-5]])
                    elif 'minus' in key:
                        d_mol[key] = np.array(minus_data[key[:-6]])
                    else:
                        d_mol[key] = np.array(base_data[key])
                    d_constraints[key] = round(sum(d_mol[key]))
                elif key == 'scf_levels':
                    scf_levels = base_data[key]
                    v = self.get_levels(scf_levels)
                    for i in v.keys():
                        d_mol[key+'_'+i] = v[i]
                elif key == 'HL_gaps':
                    scf_levels = base_data['scf_levels']
                    v = self.get_levels(scf_levels)
                    for i in v.keys():
                        if "HOMO" in i:
                            for j in v.keys():
                                if "LUMO" in j:
                                    d_mol[i+'/'+j] = v[j] - v[i]
                elif key == 'CSPA':
                    scf_levels = base_data['scf_levels']
                    CSPA = base_data['CSPA']
                    v = self.get_CSPA(scf_levels, CSPA)
                    for i in v.keys():
                        d_mol[i] = v[i]
                        d_constraints[i] = round(sum(d_mol[i]))
                elif key == 'IP':
                    d_mol[key] = plus_data['energy'] - base_data['energy']
                elif key == 'EA':
                    d_mol[key] = base_data['energy'] - minus_data['energy']
                elif key == 'smiles':
                    v = base_data['smi']
                    d_mol[key] = v
                else:
                    try:
                        v = base_data[key]
                        d_mol[key] = v
                    except:
                        raise ValueError(
                            f'Unrecognized property of {key} is provided!')

            df_dict = pd.DataFrame([d_mol])
            df_mols = pd.concat([df_mols, df_dict], ignore_index=True)

            df_constraints_dict = pd.DataFrame([d_constraints])
            df_constraints = pd.concat(
                [df_constraints, df_constraints_dict], ignore_index=True)

        # Save
        basename = os.path.splitext(output_path)[0]
        df_mols_copy = df_mols.copy()
        for key in df_mols_copy.columns:
            for i in range(len(df_mols_copy)):
                try:
                    df_mols_copy[key][i] = df_mols_copy[key][i].tolist()
                except:
                    pass
        df_mols_copy.to_csv(output_path, index=False)
        df_constraints.to_csv(f'{basename}_constraints.csv', index=False)

    def load_file(self, file):
        with open(file, 'r') as json_file:
            df = json.load(json_file)
        return df

    # Atomic Properties
    def get_elec_config(self, config):
        accepted_orbitals = ['1s', '2s', '2p', '3s',
                             '3p', '3d', '4s', '4p', '4d', '5p', '5s']
        result = []
        for line in config:
            data = line.replace('(', ' ').replace(')', ' ').split()
            info = {}
            for i in range(len(data)//2):
                info[data[2*i]] = float(data[2*i+1])
            result.append(info)
            if set(info.keys()).difference(set(accepted_orbitals)) != set():
                raise ValueError('Unsorported orbital!')
            for orbital in accepted_orbitals:
                if orbital not in info.keys():
                    if orbital == '1s':
                        info[orbital] = 2
                    else:
                        info[orbital] = 0
        _1s, _2s, _2p, _3s, _3p, _3d, _4s, _4p, _4d, _5p, _5s = [
        ], [], [], [], [], [], [], [], [], [], []
        for line in result:
            _1s.append(line['1s'])
            _2s.append(line['2s'])
            _2p.append(line['2p'])
            _3s.append(line['3s'])
            _3p.append(line['3p'])
            _3d.append(line['3d'])
            _4s.append(line['4s'])
            _4p.append(line['4p'])
            _4d.append(line['4d'])
            _5p.append(line['5p'])
            _5s.append(line['5s'])
        return {'1s': _1s, '2s': _2s, '2p': _2p, '3s': _3s, '3p': _3p, '3d': _3d, '4s': _4s, '4p': _4p, '4d': _4d, '5p': _5p, '5s': _5s}

    def get_shielding_constants(self, nmr_lewis_matrix, nmr_non_lewis_matrix):
        nmr = np.array(nmr_lewis_matrix) + np.array(nmr_non_lewis_matrix)
        nmr = np.sum(nmr, axis=0).round(2).tolist()
        return nmr

    # Bond Properties
    def get_nbo_lewis_orbital_bd(self, smi, nbo_lewis_full_orbital_bd):
        mol = make_mol(smi)
        n_atoms = len(mol.GetAtoms())
        natural_ionicity_list, s_p_list, p_p_list, d_p_list = [
            np.zeros((n_atoms, n_atoms), np.float64).tolist()] * 4  # [], [], [], []
        # Get bonds
        bonds = []
        for bond in mol.GetBonds():
            bonds.append([bond.GetBeginAtom().GetIdx() +
                         1, bond.GetEndAtom().GetIdx()+1])

        for b in bonds:
            natural_ionicity, s_p, p_p, d_p = None, None, None, None
            for line in nbo_lewis_full_orbital_bd:
                if (b[0] == line[0][4] and b[1] == line[0][6]) or (b[0] == line[0][6] and b[1] == line[0][4]):
                    natural_ionicity = line[0][8]
                    bond_char = line[0][9][1]
                    for bc in bond_char:
                        if bc[0] == 's%':
                            s_p = bc[1]
                        elif bc[0] == 'p%':
                            p_p = bc[1]
                        elif bc[0] == 'd%':
                            d_p = bc[1]
                        else:
                            raise ValueError('Unrecognized orbital!')
            if natural_ionicity is None:
                natural_ionicity, s_p, p_p, d_p = 0, 0, 0, 0

            natural_ionicity_list[b[0]-1][b[1]-1] = natural_ionicity
            natural_ionicity_list[b[1]-1][b[0]-1] = natural_ionicity
            s_p_list[b[0]-1][b[1]-1] = s_p
            s_p_list[b[1]-1][b[0]-1] = s_p
            p_p_list[b[0]-1][b[1]-1] = p_p
            p_p_list[b[1]-1][b[0]-1] = p_p
            d_p_list[b[0]-1][b[1]-1] = d_p
            d_p_list[b[1]-1][b[0]-1] = d_p

        return {'natural_ionicity': natural_ionicity_list, 's%': s_p_list, 'p%': p_p_list, 'd%': d_p_list}

    # Molecule Properties
    def get_levels(self, scf_levels):
        """
        Return +3/-3 scf levels referenced to zero.
        """
        for ind, level in enumerate(scf_levels):
            if level > 0:
                break
        return {'HOMO-3': scf_levels[ind-4], 'HOMO-2': scf_levels[ind-3], 'HOMO-1': scf_levels[ind-2], 'HOMO': scf_levels[ind-1], 'LUMO': scf_levels[ind], 'LUMO+1': scf_levels[ind+1], 'LUMO+2': scf_levels[ind+2], 'LUMO+3': scf_levels[ind+3]}

    def get_CSPA(self, scf_levels, CSPA):
        """
        Return atomic contributions to MOs (+3/-3 scf levels referenced to zero).
        """
        for ind, level in enumerate(scf_levels):
            if level > 0:
                break
        return {'HOMO-3_CSPA': CSPA[ind-4], 'HOMO-2_CSPA': CSPA[ind-3], 'HOMO-1_CSPA': CSPA[ind-2], 'HOMO_CSPA': CSPA[ind-1], 'LUMO_CSPA': CSPA[ind], 'LUMO+1_CSPA': CSPA[ind+1], 'LUMO+2_CSPA': CSPA[ind+2], 'LUMO+3_CSPA': CSPA[ind+3]}


def make_mol(smiles, atom_mapped=False):
    if atom_mapped:
        mol = RDKitMol.FromSmiles(smiles, removeHs=False, addHs=False)
    else:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
    return mol


property_names = [
    'smiles',
    'npa_charge',
    'npa_charge_plus',
    'npa_charge_minus',
    'npa_parr_neu',
    'npa_parr_elec',
    'shielding_constants',
    '1s_val_occ',
    '2s_val_occ',
    '2p_val_occ',
    '3s_val_occ',
    '3p_val_occ',
    '4s_val_occ',
    '4p_val_occ',
    'bond_index_matrix',
    'bond_length_matrix',
    'bond_charge',
    'natural_ionicity',
    'HL_gaps',
    'mulliken_dipole_tot',
    'mulliken_quadrupoles',
    'IP',
    'EA',
]
