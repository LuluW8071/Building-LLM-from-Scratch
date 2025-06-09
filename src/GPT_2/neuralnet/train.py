import os
import torch
import lightning as L
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from lightning.fabric.loggers import TensorBoardLogger

from dataset import TokenizedCorpus, TrainDataset, ValDataset
from model import GPTModel
from utils import save_best_checkpoint, should_stop_early, set_barrier


def train(fabric, model, optimizer, scheduler, train_dataloader, val_dataloader,
          max_epochs=10, checkpoint_dir="./checkpoints", patience=10, val_interval=0.5):
    global_step = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    checkpoint_paths = []

    val_interval_batches = max(1, int(len(train_dataloader) * val_interval))

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False, disable=not fabric.is_global_zero)

        for step, (x, y) in enumerate(pbar):
            logits, loss = model(x, y)

            optimizer.zero_grad()
            fabric.backward(loss)
            optimizer.step()
            scheduler.step()

            reduced_loss = fabric.all_reduce(loss, reduce_op="mean")
            total_loss += reduced_loss.item()
            global_step += 1

            pbar.set_postfix({"loss": f"{reduced_loss.item():.4f}"})

            # Log train loss every 50 steps
            if step % 50 == 0 and fabric.is_global_zero:
                fabric.logger.log_metrics({"train_loss": reduced_loss.item()}, step=global_step)

            # Validate every val_interval_batches or at epoch end
            if (step + 1) % val_interval_batches == 0 or (step + 1) == len(train_dataloader):
                avg_val_loss = validate(fabric, model, val_dataloader)

                if fabric.is_global_zero:
                    avg_train_loss = total_loss / (step + 1)
                    fabric.logger.log_metrics({
                        "avg_train_loss": avg_train_loss,
                        "val_loss": avg_val_loss
                    }, step=global_step)
                    tqdm.write(f"[Epoch {epoch+1} Step {step+1}] Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f}")

                # Save checkpoint if validation improved
                best_val_loss, epochs_without_improvement = save_best_checkpoint(
                    fabric, model, checkpoint_dir, epoch, avg_val_loss, best_val_loss,
                    epochs_without_improvement, checkpoint_paths
                )

                # Early stopping
                if should_stop_early(epochs_without_improvement, patience, fabric):
                    tqdm.write("Early stopping triggered.")
                    return


def validate(fabric, model, dataloader):
    fabric_model = model.module if hasattr(model, "module") else model
    set_barrier(fabric)
    fabric_model.eval()

    total_loss = 0.0
    count = 0
    pbar = tqdm(dataloader, desc="Validation", leave=False, disable=not fabric.is_global_zero)

    with torch.no_grad():
        for x, y in pbar:
            _, loss = fabric_model(x, y)
            total_loss += loss.item()
            count += 1
            avg_loss = total_loss / count
            pbar.set_postfix({"avg_val_loss": f"{avg_loss:.4f}"})

    avg_val_loss = fabric.all_reduce(torch.tensor(avg_loss), reduce_op="mean").item()

    fabric_model.train()
    set_barrier(fabric)
    return avg_val_loss


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger = TensorBoardLogger("logs/", name="gpt_training")

    fabric = L.Fabric(
        accelerator=args.accelerator,
        strategy=args.dist_backend,
        devices=args.gpus,
        precision=args.precision,
        loggers=logger
    )
    fabric.launch()

    corpus = TokenizedCorpus(
        corpus_folder=args.corpus_folder,
        block_size=args.block_size,
        val_split_ratio=0.10
    )

    train_dataset = TrainDataset(corpus)
    val_dataset = ValDataset(corpus)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    model = GPTModel(
        vocab_size=corpus.vocab_size,
        block_size=args.block_size,
        n_embed=384,
        n_head=6,
        n_layer=4,
        dropout=0.1,
        device=args.accelerator
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    total_steps = len(train_loader) * args.max_epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)

    model, optimizer = fabric.setup(model, optimizer)

    train(
        fabric, model, optimizer, scheduler,
        train_loader, val_loader,
        max_epochs=args.max_epochs,
        checkpoint_dir="./checkpoints",
        patience=8,
        val_interval=args.val_interval
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GPT Model with Fabric")

    # Fabric / Device args
    parser.add_argument('-a', '--accelerator', default="auto", help="Training device")
    parser.add_argument('-g', '--gpus', default=1, type=int, help='Number of devices per node')
    parser.add_argument('-w', '--num_workers', default=4, type=int, help='Data loading workers')
    parser.add_argument('-db', '--dist_backend', default='deepspeed_stage_2', type=str, help='Distributed backend strategy')

    # Dataset
    parser.add_argument('-cp', '--corpus_folder', required=True, type=str, help='Folder with .txt training files')

    # Training Hyperparameters
    parser.add_argument('-e', '--max_epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', default=2e-4, type=float, help='Learning rate')
    parser.add_argument('--block_size', default=128, type=int, help='Block size (sequence length)')
    parser.add_argument('--precision', default='16-mixed', type=str, help='Precision setting')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to checkpoint to resume training')

    # Validation interval in fraction of epoch, e.g. 0.5 means twice per epoch
    parser.add_argument('--val_interval', default=0.5, type=float, help='Fraction of epoch to run validation (e.g. 0.5)')

    args = parser.parse_args()
    main(args)