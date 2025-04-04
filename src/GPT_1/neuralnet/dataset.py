import os
import torch 
from torch.utils.data import IterableDataset, DataLoader

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_tokenizer(corpus_folder, output_path="bpe_tokenizer.json"):
    """ Train BPE Tokenizer """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    file_paths = [os.path.join(corpus_folder, fname) for fname in os.listdir(corpus_folder) if fname.endswith(".txt")]
    tokenizer.train(file_paths, trainer)
    tokenizer.save(output_path)
    return tokenizer


class TextDataset(IterableDataset):
    """ Iterable Dataset to load massive corpus of txt files """
    def __init__(self, folder, tokenizer_path, block_size) :
        self.folder = folder
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.block_size = block_size
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")]

    def __iter__(self):
        for path in self.files:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                ids = self.tokenizer.encode(text).ids

                for i in range(0, len(ids) - self.block_size, self.block_size):
                    x = torch.tensor(ids[i:i + self.block_size], dtype=torch.long)
                    y = torch.tensor(ids[i + 1:i + 1 + self.block_size], dtype=torch.long)
                    yield x, y

