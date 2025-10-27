from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold, train_test_split
import pickle
import os.path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
import itertools
from random import sample as random_sample
from itertools import chain, repeat, islice
from collections import Counter
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from herbmind.config import *


def is_none_list(x):
    try:
        return x == [set()] * len(x)
    except ValueError as e:
        return False

def car(X):

    return sum(map(lambda x: len(x), X))

def get_collate_fn(collate, args):
    if 'herbmind' in args.model_struct:
        return collate.fn_herbmind
    elif 'kitchenette' in args.model_struct:
        return collate.fn_kitchenette
    else:
        raise

def get_train_loader(args, collate):
    if 'herbmind' in args.model_struct:
        dataset = HerbMindDataset(args, 'train')
        C = collate.fn_herbmind
    elif args.model_struct == 'kitchenette':
        dataset = KitchenetteDataset(args, 'train')
        C = collate.fn_kitchenette
    else:
        raise

    return DataLoader(dataset,batch_size=args.batch_size,collate_fn=C,shuffle=True)

def get_valid_loader(args, collate):
    args.dataset_name = ''.join(args.dataset_name.split('_frac_')[0])
    if 'herbmind' in args.model_struct:
        dataset = HerbMindDataset(args, 'valid')
        C = collate.fn_herbmind
    elif args.model_struct == 'kitchenette':
        dataset = KitchenetteDataset(args, 'valid')
        C = collate.fn_kitchenette
    else:
        raise

    if args.model_struct == 'kitchenette': 
        batch_size = 128
    else:
        batch_size = 5000

    return DataLoader(dataset,batch_size=batch_size,collate_fn=C,shuffle=True)

def get_test_loader(args, collate):
    args.dataset_name = ''.join(args.dataset_name.split('_frac_')[0])
    if 'herbmind' in args.model_struct:
        dataset = HerbMindDataset(args, 'test')
        C = collate.fn_herbmind
    elif args.model_struct == 'kitchenette':
        dataset = KitchenetteDataset(args, 'test')
        C = collate.fn_kitchenette
    else:
        raise

    if args.model_struct == 'kitchenette': 
        batch_size = 128
    else:
        batch_size = 100000

    return DataLoader(dataset,batch_size=batch_size,collate_fn=C,shuffle=False)


class HerbMindData(object):
    def __init__(self, rid, entries):
        self.rid = rid
        self.xQ, self.xA, self.yS = set(), None, 0.0
        self.__dict__.update(**entries)


class KitchenetteData(object):
    def __init__(self, rid, entries):
        self.rid = rid
        self.xQ, self.xA, self.yS = set(), None, 0.0
        self.__dict__.update(**entries)
        self.REMOVE_FLAG = False
        assert len(self.xQ) == 1
        self.xQ = list(self.xQ)[0]
        if '<T>/' in self.xQ or '<T>/' in self.xA:
            self.REMOVE_FLAG = True
        self.pair = frozenset([self.xQ, self.xA])

    def __eq__(self, other): 
        if not isinstance(other, KitchenetteData):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.pair == other.pair

    def __hash__(self):
        return hash(self.pair)


class RecipeBowlData(object):
    def __init__(self, rid, entries):
        self.rid = rid
        self.xJ, self.xT, self.yJ, self.yR = None, None, None, None
        self.__dict__.update(**entries)

class HerbMindDataset(Dataset):
    def __init__(self, args=None, label='train'):
        DATA_VERSION = args.dataset_version
        DATASET_NAME = args.dataset_name
        DATASET_INDEX = args.dataset_index
        DATASET_PKL = f'{ROOT_PATH}{DATASET_INDEX}/{label}_{DATASET_NAME}.pkl'
        print("Loading Dataset      :", DATASET_PKL)
        self.dataset = pickle.load(open(DATASET_PKL, 'rb'))
        self.data_ids = [i for i in range(len(self.dataset))] 
        def build_sample(x):
            return HerbMindData(x,entries=self.dataset[x])
        self.dataset = list(map(lambda x: build_sample(x), self.data_ids))
        print("Dataset Size         :", len(self.dataset))

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class KitchenetteDataset(Dataset):
    def __init__(self, args=None, label='train'):
        DATA_VERSION = args.dataset_version
        DATASET_NAME = args.dataset_name
        DATASET_INDEX = args.dataset_index
        if 'herbmind_doublets' not in DATASET_NAME:
            DATASET_NAME = 'herbmind_doublets'
        DATASET_PKL = f'{ROOT_PATH}{DATASET_INDEX}/{label}_{DATASET_NAME}.pkl'
        print("Loading Dataset      :", DATASET_PKL)
        self.dataset = pickle.load(open(DATASET_PKL, 'rb'))
        self.data_ids = [i for i in range(len(self.dataset))] 
        def build_sample(x):
            return KitchenetteData(x,entries=self.dataset[x])
        self.dataset = list(map(lambda x: build_sample(x), self.data_ids))
        self.dataset = list(set(self.dataset))
        self.dataset = [x for x in self.dataset if not x.REMOVE_FLAG]
        print("Dataset Size         :", len(self.dataset))

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class HerbMindDataBatch(object):
    def __init__(self, *samples):
        self.batch_size = len(samples)

        def pad_list_elements(list_elements):
            def pad_infinite(iterable, padding=None):
                return chain(iterable, repeat(padding))
            def pad(iterable, size, padding=None):
                return islice(pad_infinite(iterable, padding), size)
            max_length = max([len(i) for i in list_elements])
            padded = [list(pad(elements, max_length,'[PAD]')) for elements in list_elements]
            masks = [[True if e != '[PAD]' else False for e in elems] for elems in padded]
            return padded, masks

        def get_ingreds(list_elements):
            return [x for x in list_elements if x.startswith('<J>/')]

        def get_tags(list_elements):
            return [x for x in list_elements if x.startswith('<T>/')]

        self.rid_list = [s.rid for s in samples]
        self.uS = [(s.xQ, s.xA, s.yS) for s in samples]
        self.xQ = [list(s.xQ) for s in samples]
        self.xA = [s.xA for s in samples]
        self.yS = [s.yS for s in samples]

        self.mQ = []
        if car(self.xQ) > 0:
            self.xQ, self.mQ = pad_list_elements(self.xQ)

    def __len__(self):

        return self.batch_size


class KitchenetteDataBatch(object):
    def __init__(self, *samples):
        self.batch_size = len(samples)

        self.rid_list = [s.rid for s in samples]
        self.uS = [(s.xQ, s.xA, s.yS) for s in samples]
        self.xQ = [s.xQ for s in samples]
        self.xA = [s.xA for s in samples]
        self.yS = [s.yS for s in samples]

    def __len__(self):

        return self.batch_size


class IngredientIterator:
    def __init__(self, list_herbs):
        self.list_herbs = list_herbs

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx <= len(self.list_herbs):
            x = self.idx
            self.idx += 1
            return '<J>/'+self.list_herbs[x]
        else:
            self.idx = 0
            return '<J>/'+self.list_herbs[x]


class CollateFn(object):
    def __init__(self, args):
        self.ingred2vec = pickle.load(open(f'{ROOT_PATH}{args.initial_vectors_J}_J_{args.dataset_version}.pkl', 'rb'))
        self.eval_prescription_completion = False 
        self.eval_food_pairing = False
        self.eval_ntuplet_scoring = False

    def modify_batch_tensor(self, ingred, batch_tensor):
        def vec(x):
            if x.startswith('<J>'):
                return self.ingred2vec[x.split('/')[1]]
            # elif x.startswith('<T>'):
            #     return self.tag2vec[x.split('/')[1]]
            elif x == '[PAD]':
                return self.ingred2vec['[PAD]']
            else:
                raise
        batch_xA = np.vstack([vec(ingred) for _ in range(batch_tensor['bs'])])
        batch_tensor['xA'] = Variable(torch.cuda.FloatTensor(batch_xA))

        return batch_tensor

    def fn_herbmind(self, batch):
        if not self.eval_prescription_completion:
            batch = HerbMindDataBatch(*batch)  
        else:
            batch = HerbMindCompletionDataBatch(*batch)

        def get_word_vector(x):
            if x.startswith('<J>'):
                return self.ingred2vec[x.split('/')[1]]
            elif x == '[PAD]':
                return self.ingred2vec['[PAD]']
            else:
                print(x); raise

        def get_ingred_vector(x):
            if x.startswith('<J>'):
                return self.ingred2vec[x.split('/')[1]]
            elif x == '[PAD]':
                return self.ingred2vec['[PAD]']
            else:
                print(x); raise

        def func(x_list, vec):
            if len(x_list) == 0: x_list.append('[PAD]')
            y = np.vstack([vec(x) for x in x_list])
            y = torch.cuda.FloatTensor(y).unsqueeze(0)
            return y   
            
        batch_tensor = dict()
        batch_tensor['rid'] = batch.rid_list
        batch_tensor['bs'] = len(batch.rid_list)
        batch_tensor['uS'] = batch.uS     
        batch_tensor['xQ'] = Variable(torch.cat(list(map(lambda x: func(x, get_word_vector), batch.xQ)), 0))     
        batch_tensor['mQ'] = torch.cuda.FloatTensor(batch.mQ)


        if not self.eval_prescription_completion:
            batch_xA = np.vstack([get_ingred_vector(x) for x in batch.xA])
            batch_tensor['xA'] = Variable(torch.cuda.FloatTensor(batch_xA))
            batch_tensor['yS'] = Variable(torch.cuda.FloatTensor(batch.yS).view(-1,1))

        return batch_tensor

    def fn_kitchenette(self, batch):
        batch = KitchenetteDataBatch(*batch)

        def vec(x):
            return self.ingred2vec[x.split('/')[1]]

        def func(x_list):
            y = np.vstack([vec(x) for x in x_list])
            y = torch.cuda.FloatTensor(y).unsqueeze(0)
            return y   

        batch_tensor = dict()
        batch_tensor['rid'] = batch.rid_list
        batch_tensor['bs'] = len(batch.rid_list)
        batch_tensor['uS'] = batch.uS
        batch_xQ = np.vstack([vec(x) for x in batch.xQ])
        batch_tensor['xQ'] = Variable(torch.cuda.FloatTensor(batch_xQ))
        batch_xA = np.vstack([vec(x) for x in batch.xA])
        batch_tensor['xA'] = Variable(torch.cuda.FloatTensor(batch_xA))
        batch_tensor['yS'] = Variable(torch.cuda.FloatTensor(batch.yS).view(-1,1))

        return batch_tensor

    def fn_inference(self, batch):
        self.idx2ingred = sorted(list(self.ingred2vec.keys()))

        if len(batch) < 1: return
        batch = RecipeBowlDataBatch(*batch)
        def func(x_list):
            y = np.zeros(len(self.idx2ingred))
            for x in x_list:
                if x != '[PAD]':
                    y[self.idx2ingred.index(x)] = 1
            y = y.reshape(1,-1)
            y = torch.cuda.FloatTensor(y)
            return y

        batch_tensor = dict()
        batch_tensor['rid'] = batch.rid_list
        batch_tensor['bs'] = len(batch.rid_list)
        batch_tensor['uS'] = batch.uS
        batch_tensor['xJ'] = torch.cat([func(x) for x in batch.xJ] )
        batch_tensor['yJ'] = torch.cuda.FloatTensor([self.idx2ingred.index(x) for x in batch.yJ])
        batch_tensor['yX'] = torch.cat([func([x]) for x in batch.yJ])

        return batch_tensor




def numpify(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor 
    else:
        raise


class StepwiseDataset:
    """Utility dataset for interactive stepwise recommendation analysis."""

    def __init__(self, args):
        self.data_dir = getattr(args, 'data_dir', './data/')
        self.device = getattr(args, 'device', torch.device('cpu'))

        self.prescriptions = self._load_prescriptions(self.data_dir)
        all_herbs = sorted({herb for herbs in self.prescriptions.values() for herb in herbs})
        self.item2idx = {herb: idx for idx, herb in enumerate(all_herbs)}
        self.idx2item = {idx: herb for herb, idx in self.item2idx.items()}
        self.item_ids = [self.idx2item[i] for i in range(len(self.idx2item))]
        self.n_items = len(self.item2idx)

        herb_counts = Counter(herb for herbs in self.prescriptions.values() for herb in herbs)
        self.herb_frequency = herb_counts

    @staticmethod
    def _guess_cols(df):
        id_col, herb_col = None, None
        for col in df.columns:
            col_upper = str(col).upper()
            if id_col is None and ('처방' in str(col) or 'ID' in col_upper):
                id_col = col
            if herb_col is None and ('약재' in str(col) or 'HERB' in col_upper):
                herb_col = col
        return id_col, herb_col

    def _load_prescriptions(self, data_dir):
        prescriptions = {}
        if not os.path.isdir(data_dir):
            return prescriptions

        for fname in os.listdir(data_dir):
            if not fname.lower().endswith('.csv'):
                continue
            path = os.path.join(data_dir, fname)
            try:
                df = pd.read_csv(path, encoding='utf-8-sig')
            except Exception:
                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue
            id_col, herb_col = self._guess_cols(df)
            if not id_col or not herb_col:
                continue
            for pid, group in df.groupby(id_col):
                herbs = [str(x).strip() for x in group[herb_col].dropna() if str(x).strip()]
                if herbs:
                    prescriptions[str(pid)] = list(dict.fromkeys(herbs))
        return prescriptions

    def get_inference_batch(self, query_set, candidate=None):
        query_list = [herb for herb in query_set if herb in self.item2idx]
        if not query_list:
            raise ValueError('Query set must contain at least one known herb.')

        query_indices = [self.item2idx[h] for h in query_list]
        seq_len = len(query_indices)
        xQ = torch.zeros((1, seq_len, self.n_items), dtype=torch.float32, device=self.device)
        for pos, idx in enumerate(query_indices):
            xQ[0, pos, idx] = 1.0

        mQ = torch.ones((1, seq_len), dtype=torch.float32, device=self.device)
        xA = torch.zeros((1, self.n_items), dtype=torch.float32, device=self.device)
        if candidate is not None and candidate in self.item2idx:
            cand_idx = self.item2idx[candidate]
            xA[0, cand_idx] = 1.0

        batch = {
            'xQ': xQ,
            'mQ': mQ,
            'xA': xA,
            'bs': 1,
            'item_ids': self.item_ids,
            'query_indices': query_indices,
            'candidate': candidate,
        }
        return batch
