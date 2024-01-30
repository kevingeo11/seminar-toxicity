import numpy as np
import pandas as pd
import os

from rdkit import Chem
import mordred
from mordred import Calculator, descriptors

from tqdm import tqdm

class mordredWrapper:
    def __init__(self, smiles_array) -> None:
        self.calc = Calculator(descriptors, ignore_3D=True)

        mol_list = []
        for smiles in smiles_array:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol_list.append(mol)

        df_mordred = self.calc.pandas(mol_list)
        truth_map = df_mordred.applymap(lambda x : not isinstance(x, mordred.error.MissingValueBase))
        truth_series = truth_map.all(axis=0)
        self.mask = truth_series.to_numpy()

    def get_fingerprints(self, smiles_array, labels):
        fps = []
        y = []
        for smiles, label in zip(smiles_array, labels):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                pass
            else:
                fps.append(np.array(self.calc(mol))[self.mask])
                y.append(label)

        assert len(fps) == len(y)
        
        return np.array(fps), np.array(y)
