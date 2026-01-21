import torch

from rdkit import Chem
from rdkit.Chem import AllChem


def get_mol(smiles):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def get_mol_conformers(n_confs, mol, target_torsions=None):
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs)

    n_atoms = mol.GetNumAtoms()

    if target_torsions is not None:
        for conf_id in conf_ids:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            for (i, j, k, l, angle) in target_torsions:
                ff.UFFAddTorsionConstraint(
                    i, j, k, l, relative=False, minDihedralDeg=float(angle),
                    maxDihedralDeg=float(angle), forceConstant=100.0
                )
            ff.Minimize()

    for conf_id in conf_ids:
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id)

    conformers = []
    for conf_id in conf_ids:
        conf = mol.GetConformer(conf_id)
        coords = []
        for atom_ind in range(n_atoms):
            pos = conf.GetAtomPosition(atom_ind)
            coords.append([pos.x, pos.y, pos.z])
        conformers.append(coords)

    conformers = torch.tensor(conformers, dtype=torch.float32)
    assert conformers.shape == (n_confs, n_atoms, 3)

    return conformers


def get_Z_Q(mol):
    Z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    Q = [atom.GetFormalCharge() for atom in mol.GetAtoms()]

    return Z, Q


def get_bonds(mol):
    bonds = []
    for bond in mol.GetBonds():
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    return bonds


def get_angles(mol):
    angles = []
    for atom in mol.GetAtoms():
        j = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]

        for a in range(len(neighbors)):
            for b in range(a+1, len(neighbors)):
                i = neighbors[a]
                k = neighbors[b]
                angles.append((i, j, k))

    return angles


def get_heavy_atom_dihedrals(mol, heavy_atom_num):
    dihedrals = []
    for bond in mol.GetBonds():
        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()

        atom_j = mol.GetAtomWithIdx(j)
        atom_k = mol.GetAtomWithIdx(k)

        if atom_j.GetAtomicNum() not in heavy_atom_num:
            continue

        if atom_k.GetAtomicNum() not in heavy_atom_num:
            continue

        for atom_i in atom_j.GetNeighbors():
            i = atom_i.GetIdx()
            if i == k:
                continue

            if atom_i.GetAtomicNum() not in heavy_atom_num:
                continue

            for atom_l in atom_k.GetNeighbors():
                l = atom_l.GetIdx()
                if l == j:
                    continue

                if atom_l.GetAtomicNum() not in heavy_atom_num:
                    continue

                dihedrals.append((i, j, k, l))

    return dihedrals
