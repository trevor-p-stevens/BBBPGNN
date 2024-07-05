#%% Create class
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset
import numpy as np 
import os
from tqdm import tqdm
import deepchem as dc
from rdkit import Chem 

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class BBBPDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        self.test = test
        self.filename = filename
        super(BBBPDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [self.filename]
    
    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
        
    def download(self):
        pass

    def process(self):
        self.skip_list = []
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        ftzr = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        for index,row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                print(f"bad smiles at index {index}: {row['smiles']}")
                self.skip_list.append(index)
                continue

            fts = ftzr._featurize(mol)
            data = fts.to_pyg_graph()
            data.y = self._get_label(row["p_np"])
            data.smiles = row["smiles"]
            if self.test:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_test_{index-len(self.skip_list)}.pt'))
            else:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_{index-len(self.skip_list)}.pt'))
    
    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)
    
    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'))

        else:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_{idx}.pt'))
        return data

    def len(self):
        #return self.data.shape[0] 
        return 2517  
    
    
    


# %% Load dataset

dataset = BBBPDataset(root='data/', filename='BBBP_train_over.csv')

# %%

print(dataset[1])
# %%
