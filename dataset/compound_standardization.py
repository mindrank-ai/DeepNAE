import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem.Scaffolds import MurckoScaffold
from joblib import Parallel, delayed


def gen_mol(smi):
    try:
        return Chem.MolFromSmiles(smi)
    except:
        return None


def gen_mol_feature(smi_list,num_jobs):
    mols = []
    
    features_map = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(gen_mol)(smi) for smi in tqdm(smi_list)
    )
    for i, feats in enumerate(features_map):
        mols.append(feats)
    return mols


def smi_to_smi(x):
    try:
        mol = Chem.MolFromSmiles(x)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
        return Chem.MolToSmiles(mol)
    except:
        return np.nan
    
def smiles_check(df, smi_col,jobs):
    '''
    Args:
        df: DataFrame
        smi_col: SMILES column name, str
        num_jobs: Number of worker, str
    returns:
        df: DataFrame with a new column named 'Smiles_check'
        print(df['Smiles_check'].value_counts(dropna=False)): 
        if all values are SANITIZE_NONE means all SMILES are right, if False come out means corresponding SMILES are wrong which need furthur amend
    '''
    mols =  gen_mol_feature(df[smi_col].values,num_jobs=jobs)
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
    '''
    Args:
        df: DataFrame
        smi_col: SMILES column name, str
    returns:
        df: DataFrame with a new column named 'Smiles_removesalt': a col contains SMILES without any salt/solvent
    '''
    Smiles_rs = []
    for smiles in tqdm(df[smi_col].values):
        frags = smiles.split(".")
        frags = sorted(frags, key=lambda x: len(x), reverse=True)
        Smiles_rs.append(frags[0])
            
    df['Smiles_removesalt'] = Smiles_rs
    return df



def smiles_unify(df, smi_col,jobs):
    '''
    Args:
        df: DataFrame
        smi_col: SMILES column name, str
    returns:
        df: DataFrame with a new column named 'Smiles_unify': a col contains unified SMILES
    '''
    Smiles_unify = []
    mols =  gen_mol_feature(df[smi_col].values,num_jobs=jobs)
    for m in tqdm(mols):
        s_u = Chem.MolToSmiles(m)
        Smiles_unify.append(s_u)
    
    df['Smiles_unify'] = Smiles_unify
    return df

