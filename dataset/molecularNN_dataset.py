from joblib import Parallel, delayed
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors, QED
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
from rdkit.Chem.Scaffolds import MurckoScaffold


def count_hydrogens(molecule):
    """Calculate the total number of hydrogen atoms in the molecule."""
    return sum(1 for atom in Chem.AddHs(molecule).GetAtoms() if atom.GetAtomicNum() == 1)

def count_halogens(molecule):
    """Calculate the number of halogen atoms (F, Cl, Br, I) in the molecule."""
    halogen_atomic_numbers = {9, 17, 35, 53}
    return sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() in halogen_atomic_numbers)

def count_aromatic_bonds(molecule):
    """Calculate the total number of aromatic bonds in the molecule."""
    return sum(1 for bond in molecule.GetBonds() if bond.GetBondType().name == 'AROMATIC')

def count_total_atoms(molecule):
    """Calculate the total number of atoms in the molecule, including hydrogens."""
    return Chem.AddHs(molecule).GetNumAtoms()

def compute_csp3_carbon_count(molecule):
    """Calculate the number of sp3 hybridized carbons in the molecule."""
    sp3_fraction = Chem.rdMolDescriptors.CalcFractionCSP3(molecule)
    total_carbon_atoms = sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6)
    return total_carbon_atoms * sp3_fraction

def count_aromatic_nitrogen_containing_rings(molecule):
    """Calculate the number of aromatic rings containing nitrogen atoms in the molecule."""
    ring_info = molecule.GetRingInfo().BondRings()
    ring_count = 0
    for ring in ring_info:
        is_aromatic = all(molecule.GetBondWithIdx(bond_idx).GetBondType().name == 'AROMATIC' for bond_idx in ring)
        contains_nitrogen = any(molecule.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomicNum() == 7 or 
                                molecule.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomicNum() == 7 for bond_idx in ring)
        if is_aromatic and contains_nitrogen:
            ring_count += 1
    return ring_count

def count_unsaturated_nonaromatic_carbocyclic_rings(molecule):
    """Calculate the number of unsaturated, non-aromatic carbocyclic rings in the molecule."""
    ring_info = molecule.GetRingInfo().BondRings()
    ring_count = 0
    for ring in ring_info:
        has_unsaturation = any(molecule.GetBondWithIdx(bond_idx).GetBondType().name != 'SINGLE' for bond_idx in ring)
        is_nonaromatic = all(molecule.GetBondWithIdx(bond_idx).GetBondType().name != 'AROMATIC' for bond_idx in ring)
        is_carbocycle = all(molecule.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomicNum() == 6 and 
                            molecule.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomicNum() == 6 for bond_idx in ring)
        if has_unsaturation and is_nonaromatic and is_carbocycle:
            ring_count += 1
    return ring_count

def count_unsaturated_nonaromatic_nitrogen_rings(molecule):
    """Calculate the number of unsaturated, non-aromatic rings containing nitrogen atoms in the molecule."""
    ring_info = molecule.GetRingInfo().BondRings()
    ring_count = 0
    for ring in ring_info:
        has_unsaturation = any(molecule.GetBondWithIdx(bond_idx).GetBondType().name != 'SINGLE' for bond_idx in ring)
        is_nonaromatic = all(molecule.GetBondWithIdx(bond_idx).GetBondType().name != 'AROMATIC' for bond_idx in ring)
        contains_nitrogen = any(molecule.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomicNum() == 7 or 
                                molecule.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomicNum() == 7 for bond_idx in ring)
        if has_unsaturation and is_nonaromatic and contains_nitrogen:
            ring_count += 1
    return ring_count

def count_unsaturated_nonaromatic_heterocyclic_rings(molecule):
    """Calculate the number of unsaturated, non-aromatic rings containing heteroatoms in the molecule."""
    ring_info = molecule.GetRingInfo().BondRings()
    ring_count = 0
    for ring in ring_info:
        has_unsaturation = any(molecule.GetBondWithIdx(bond_idx).GetBondType().name != 'SINGLE' for bond_idx in ring)
        is_nonaromatic = all(molecule.GetBondWithIdx(bond_idx).GetBondType().name != 'AROMATIC' for bond_idx in ring)
        contains_heteroatom = any(molecule.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomicNum() not in {1, 6} or 
                                  molecule.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomicNum() not in {1, 6} for bond_idx in ring)
        if has_unsaturation and is_nonaromatic and contains_heteroatom:
            ring_count += 1
    return ring_count

def gen_qed_properties(m):
    """Generate QED properties for the given molecule."""
    qed_properties = QED.properties(m)
    return {
        'QED_ALOGP': qed_properties.ALOGP,
        'QED_MW': qed_properties.MW,
        'QED_ROTB': qed_properties.ROTB,
        'QED_HBA': qed_properties.HBA,
        'QED_HBD': qed_properties.HBD,
        'QED_PSA': qed_properties.PSA,
        'QED_AROM': qed_properties.AROM,
    }

descriptor_functions = {
    'HydrogenAtomCount': count_hydrogens,
    'HalogenAtomCount': count_halogens,
    'AromaticBondCount': count_aromatic_bonds,
    'TotalAtomCount': count_total_atoms,
    'Sp3CarbonCount': compute_csp3_carbon_count,
    'AromaticNitrogenRingCount': count_aromatic_nitrogen_containing_rings,
    'UnsaturatedNonaromaticCarbocyclicRingCount': count_unsaturated_nonaromatic_carbocyclic_rings,
    'UnsaturatedNonaromaticNitrogenRingCount': count_unsaturated_nonaromatic_nitrogen_rings,
    'UnsaturatedNonaromaticHeterocyclicRingCount': count_unsaturated_nonaromatic_heterocyclic_rings,
    'HeteroatomCount': rdMolDescriptors.CalcNumHeteroatoms,
    'RingCount': rdMolDescriptors.CalcNumRings,
    'AmideBondCount': rdMolDescriptors.CalcNumAmideBonds,
    'QED_ALOGP': lambda m: gen_qed_properties(m)['QED_ALOGP'],
    'QED_MW': lambda m: gen_qed_properties(m)['QED_MW'],
    'QED_ROTB': lambda m: gen_qed_properties(m)['QED_ROTB'],
    'QED_HBA': lambda m: gen_qed_properties(m)['QED_HBA'],
    'QED_HBD': lambda m: gen_qed_properties(m)['QED_HBD'],
    'QED_PSA': lambda m: gen_qed_properties(m)['QED_PSA'],
    'QED_PSA': lambda m: gen_qed_properties(m)['QED_PSA'],
    'QED_AROM': lambda m: gen_qed_properties(m)['QED_AROM'],
}

def gen_mogan(m, radius=2, nBits=2048):
    try:
        MorganGenerator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits, includeChirality=True)
        fp = MorganGenerator.GetFingerprint(m)
        return np.array(fp, dtype=np.int64)
    except Exception as e:
        print(f"Error generating fingerprint: {e}")
        return None

def gen_morgan_feature(mol_list, num_jobs):
    features_map = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(gen_mogan)(mol) for mol in tqdm(mol_list)
    )
    return np.array(features_map)

def get_molecular_descriptors(molecule):
    """Get a dictionary of molecular descriptors for the given molecule."""
    descriptor_values = OrderedDict()
    for descriptor_name, function in descriptor_functions.items():
        try:
            descriptor_values[descriptor_name] = function(molecule)
        except:
            descriptor_values[descriptor_name] = 0.0
    return list(descriptor_values.values())

def gen_descriptors_feature(mol_list, num_jobs):
    features_map = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(get_molecular_descriptors)(mol) for mol in tqdm(mol_list)
    )
    return np.array(features_map)

def gen_mol(smi):
    try:
        return Chem.MolFromSmiles(smi)
    except:
        return None

def gen_mol_feature(smi_list, num_jobs):
    features_map = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(gen_mol)(smi) for smi in tqdm(smi_list)
    )
    return features_map

def smi_to_smi(x):
    try:
        mol = Chem.MolFromSmiles(x)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
        return Chem.MolToSmiles(mol)
    except:
        return np.nan

def smiles_check(df, smi_col, jobs):
    """
    Args:
        df: DataFrame
        smi_col: SMILES column name, str
        jobs: Number of worker, int
    Returns:
        df: DataFrame with a new column named 'Smiles_check'
    """
    mols = gen_mol_feature(df[smi_col].values, num_jobs=jobs)
    check_result = []
    for mol in tqdm(mols):
        a = bool(mol)
        if a:
            b = str(Chem.SanitizeMol(mol))
            check_result.append(b)
        else:
            check_result.append(a)

    df['Smiles_check'] = check_result
    print(df['Smiles_check'].value_counts(dropna=False))
    return df

def remove_salt(df, smi_col):
    """
    Args:
        df: DataFrame
        smi_col: SMILES column name, str
    Returns:
        df: DataFrame with a new column named 'Smiles_removesalt': a col contains SMILES without any salt/solvent
    """
    Smiles_rs = []
    for smiles in tqdm(df[smi_col].values):
        frags = smiles.split(".")
        frags = sorted(frags, key=lambda x: len(x), reverse=True)
        Smiles_rs.append(frags[0])
            
    df['Smiles_removesalt'] = Smiles_rs
    return df

def smiles_unify(df, smi_col, jobs):
    """
    Args:
        df: DataFrame
        smi_col: SMILES column name, str
    Returns:
        df: DataFrame with a new column named 'Smiles_unify': a col contains unified SMILES
    """
    Smiles_unify = []
    mols = gen_mol_feature(df[smi_col].values, num_jobs=jobs)
    for m in tqdm(mols):
        s_u = Chem.MolToSmiles(m)
        Smiles_unify.append(s_u)
    
    df['Smiles_unify'] = Smiles_unify
    return df

class Molecule:
    def __init__(self, mol, label, cpu_core=48):
        self.mol = mol
        self.label = label
        self.fingerprints = gen_mogan(self.mol)
        self.features = get_molecular_descriptors(self.mol)

class MolDataSet(Dataset):
    def __init__(self, data_list):
        self.data_list = np.array(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return MolDataSet(self.data_list[key])
        return self.data_list[key]

def construct_molecularNN_dataset(mol_list, label_list):
    output = [Molecule(mol, label) for mol, label in tqdm(zip(mol_list, label_list), total=len(mol_list))]
    return MolDataSet(output)

def molecularNN_mol_collate_func(batch):
    result = {
        'fingerprints': torch.from_numpy(np.array([x.fingerprints for x in batch])).float(),
        'features': torch.from_numpy(np.array([x.features for x in batch])).float(),
        'target': torch.from_numpy(np.array([x.label for x in batch])).float()
    }
    return result

def generate_moleceularNN_embeddings(mol_list,  model, logger, batch_size=1024, device='cuda:2'):
    """Generates embeddings for a list of SMILES strings."""
    all_embeddings = []
    for i in tqdm(range(0, len(mol_list), batch_size), desc="Processing Batches"):
        batch_mol = mol_list[i:i+batch_size]
        try:
            batch_fp = gen_morgan_feature(batch_mol, 48)
            batch_feature = gen_descriptors_feature(batch_mol, 48)
            batch_fp = torch.tensor(batch_fp, dtype=torch.float).to(device)
            batch_feature = torch.tensor(batch_feature, dtype=torch.float).to(device)
        except ValueError as e:
            logger.error(f'Error processing batch {batch_mol}: {e}')
            raise e
        
        with torch.no_grad():
            batch_embedding = model(batch_fp, output_embedding=True, feature=batch_feature)
        all_embeddings.append(batch_embedding.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def generate_embeddings(smiles):
    """Generates embeddings for a list of SMILES strings."""

    try:
        mol = gen_mol(smiles)
        fp = gen_mogan(mol)
    except ValueError as e:
        print(f'Error processing generate embeddings:{e}')
        raise e
        
    return fp

def generate_embeddings_features(smi_list,num_jobs):
    mols = []
    
    features_map = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(generate_embeddings)(smi) for smi in tqdm(smi_list)
    )
    for i, feats in enumerate(features_map):
        mols.append(feats)
    return mols
