import os
import random
import torch
from torch.utils.data import IterableDataset, DataLoader

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class BaseShardedDataset(IterableDataset):
    def __init__(self, corpus_folder, fabric, block_size=32, split="train", val_split_ratio=0.02, cache_dir="tokenized_cache"):
        self.rank = fabric.global_rank
        self.world_size = fabric.world_size
        self.block_size = block_size
        self.split = split
        self.val_split_ratio = val_split_ratio
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.tokenizer = self._get_or_train_tokenizer(corpus_folder)
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.files = self._get_sharded_files(corpus_folder)

    def _get_sharded_files(self, corpus_folder):
        all_files = sorted([
            os.path.join(corpus_folder, f)
            for f in os.listdir(corpus_folder)
            if f.endswith(".txt")
        ])
        return all_files[self.rank::self.world_size]

    def _get_or_train_tokenizer(self, corpus_folder, tokenizer_path="bpe_tokenizer.json"):
        if os.path.exists(tokenizer_path):
            print("Loading existing tokenizer...")
            return Tokenizer.from_file(tokenizer_path)
        else:
            print("Training new tokenizer...")
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

            file_paths = [
                os.path.join(corpus_folder, fname)
                for fname in os.listdir(corpus_folder)
                if fname.endswith(".txt")
            ]

            tokenizer.train(file_paths, trainer)
            tokenizer.save(tokenizer_path)
            return tokenizer

    def _load_or_tokenize_file(self, path):
        """ 
        Saving cache of tokenization for no redundation tokenization process
        Speeds up later epochs 
        """
        fname = os.path.basename(path).replace(".txt", ".ids.pt")
        cache_path = os.path.join(self.cache_dir, fname)

        if os.path.exists(cache_path):
            ids = torch.load(cache_path, weights_only=True)
        else:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            ids = torch.tensor(self.tokenizer.encode(text).ids, dtype=torch.long)
            torch.save(ids, cache_path)
        return ids


class TrainDataset(BaseShardedDataset):
    def __iter__(self):
        files = self.files[:]    # Copy list
        random.shuffle(files)    # Random shuffling of Training Dataset

        for path in files:
            # print(path)
            ids = self._load_or_tokenize_file(path)
            # Token Based Split for Training
            split_idx = int(len(ids) * (1 - self.val_split_ratio))
            ids = ids[:split_idx]

            for i in range(0, len(ids) - self.block_size, self.block_size):
                x = ids[i:i + self.block_size]
                y = ids[i + 1:i + 1 + self.block_size]
                yield x, y


class ValDataset(BaseShardedDataset):
    def __iter__(self):
        for path in self.files:
            ids = self._load_or_tokenize_file(path)
            # Token Based Split for Validation
            split_idx = int(len(ids) * (1 - self.val_split_ratio))
            ids = ids[split_idx:]

            for i in range(0, len(ids) - self.block_size, self.block_size):
                x = ids[i:i + self.block_size]
                y = ids[i + 1:i + 1 + self.block_size]
                yield x, y