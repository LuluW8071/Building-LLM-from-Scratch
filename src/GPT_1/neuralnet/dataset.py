import os
import torch

from typing import List, Tuple
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from filelock import FileLock


# ======================================================================================
class TokenizedCorpus:
    """
    A utility class to preprocess a folder of text files for language modeling tasks.
    This includes training or loading a Byte-Pair Encoding (BPE) tokenizer, 
    tokenizing the corpus, caching results, and preparing training and validation datasets.

    Args:
        corpus_folder (str): Path to the directory containing `.txt` files.
        block_size (int): Length of each training/validation chunk in tokens.
        val_split_ratio (float): Fraction of the data to use for validation.
        cache_dir (str): Path to cache the tokenized files.

    Attributes:
        tokenizer (Tokenizer): Trained or loaded BPE tokenizer.
        vocab_size (int): Vocabulary size from the tokenizer.
        train_chunks (List[Tuple[Tensor, Tensor]]): Training samples (input, target).
        val_chunks (List[Tuple[Tensor, Tensor]]): Validation samples (input, target).
    """

    def __init__(
            self,
            corpus_folder: str,
            block_size: int = 32,
            val_split_ratio: float = 0.10,
            cache_dir: str = "tokenized_cache"
    ):
        self.block_size = block_size
        self.val_split_ratio = val_split_ratio
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.tokenizer = self._get_or_train_tokenizer(corpus_folder)
        self.vocab_size = len(self.tokenizer.get_vocab())

        self.train_chunks, self.val_chunks = self._load_and_split_all_files(
            corpus_folder)

    def _get_or_train_tokenizer(self, corpus_folder: str, tokenizer_path: str = "bpe_tokenizer.json") -> Tokenizer:
        """
        Loads a tokenizer from file if available, otherwise trains a new BPE tokenizer.

        Args:
            corpus_folder (str): Directory containing training .txt files.
            tokenizer_path (str): Path to save/load the tokenizer.

        Returns:
            Tokenizer: A trained or loaded BPE tokenizer.
        """
        if os.path.exists(tokenizer_path):
            print("Loading existing tokenizer...")
            return Tokenizer.from_file(tokenizer_path)
        else:
            print("Training new tokenizer...")
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(vocab_size=36000,
                                 special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

            file_paths = [
                os.path.join(corpus_folder, fname)
                for fname in os.listdir(corpus_folder)
                if fname.endswith(".txt")
            ]

            tokenizer.train(file_paths, trainer)
            tokenizer.save(tokenizer_path)
            return tokenizer

    def _load_or_tokenize_file(self, path: str) -> torch.Tensor:
        """
        Loads a cached tokenized file or tokenizes and caches it.

        Args:
            path (str): Path to the .txt file.

        Returns:
            torch.Tensor: Tokenized tensor of token IDs.
        """
        fname = os.path.basename(path).replace(".txt", ".ids.pt")
        cache_path = os.path.join(self.cache_dir, fname)
        lock_path = cache_path + ".lock"

        with FileLock(lock_path):
            if os.path.exists(cache_path):
                try:
                    ids = torch.load(cache_path, weights_only=True)
                except (EOFError, RuntimeError):
                    print(
                        f"Corrupt cache detected. Deleting {cache_path} and reprocessing.")
                    os.remove(cache_path)
                    return self._load_or_tokenize_file(path)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                ids = torch.tensor(self.tokenizer.encode(
                    text).ids, dtype=torch.long)
                torch.save(ids, cache_path)

        return ids

    def _make_chunks(self, ids: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Splits token IDs into overlapping input-target chunks of block_size.

        Args:
            ids (torch.Tensor): 1D tensor of token IDs.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: List of (x, y) training samples.
        """
        chunks = []
        for i in range(0, len(ids) - self.block_size, self.block_size):
            x = ids[i:i + self.block_size]
            y = ids[i + 1:i + 1 + self.block_size]
            if len(x) == self.block_size and len(y) == self.block_size:
                chunks.append((x, y))
        return chunks

    def _load_and_split_all_files(self, corpus_folder: str) -> Tuple[
        List[Tuple[torch.Tensor, torch.Tensor]],
        List[Tuple[torch.Tensor, torch.Tensor]]
    ]:
        """
        Loads, tokenizes, and splits all text files in the corpus into train/val sets.

        Args:
            corpus_folder (str): Directory containing .txt files.

        Returns:
            Tuple: (train_chunks, val_chunks), both lists of (x, y) tuples.
        """
        train_chunks = []
        val_chunks = []

        for fname in sorted(os.listdir(corpus_folder)):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(corpus_folder, fname)
            ids = self._load_or_tokenize_file(path)

            split_idx = int(len(ids) * (1 - self.val_split_ratio))
            train_ids = ids[:split_idx]
            val_ids = ids[split_idx:]

            train_chunks.extend(self._make_chunks(train_ids))
            val_chunks.extend(self._make_chunks(val_ids))

        return train_chunks, val_chunks
# ======================================================================================


class TrainDataset(Dataset):
    """
    PyTorch Dataset wrapper for training data chunks from TokenizedCorpus.

    Args:
        corpus (TokenizedCorpus): The preprocessed tokenized corpus.
    """

    def __init__(self, corpus: TokenizedCorpus):
        self.data: List[Tuple[torch.Tensor, torch.Tensor]
                        ] = corpus.train_chunks

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


class ValDataset(Dataset):
    """
    PyTorch Dataset wrapper for validation data chunks from TokenizedCorpus.

    Args:
        corpus (TokenizedCorpus): The preprocessed tokenized corpus.
    """

    def __init__(self, corpus: TokenizedCorpus):
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = corpus.val_chunks

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]
