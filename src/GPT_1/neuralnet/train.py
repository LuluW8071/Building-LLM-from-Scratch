import os
import torch
import lightning as L
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TrainDataset, ValDataset
from model import GPTModel
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(fabric, model, optimizer, scheduler, train_dataloader, val_dataloader, max_epochs=10, val_interval=100, checkpoint_dir="./checkpoints", patience=3):
    """ Training Loop with epochs, tqdm progress bar, checkpointing, and early stopping """
    model.train()
    global_step = 0
    best_val_loss = float('inf')  # Keep track of the best validation loss
    epochs_without_improvement = 0
    checkpoint_paths = []

    for epoch in range(max_epochs):
        total_steps = 0
        total_loss_step_window = 0.0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)

        for batch_idx, (x, y) in enumerate(pbar):
            logits, loss = model(x, y)
            optimizer.zero_grad()
            fabric.backward(loss)
            optimizer.step()

            # Reduce loss across processes
            reduced_loss = fabric.all_reduce(loss.detach(), reduce_op="mean")
            step_loss = reduced_loss.item()

            total_loss_step_window += step_loss
            total_steps += 1
            global_step += 1

            if global_step % val_interval == 0:
                avg_val_loss = validate(fabric, model, val_dataloader)
                avg_train_loss = total_loss_step_window / val_interval
                total_loss_step_window = 0.0  # Reset step window

                # Checkpointing: Save model if the validation loss improves
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                    
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{global_step}_val_loss_{best_val_loss:.4f}.pt")
                    
                    torch.save(model.state_dict(), checkpoint_path)
                    checkpoint_paths.append(checkpoint_path)

                    # Keep only the top 5 checkpoints
                    if len(checkpoint_paths) > 5:
                        os.remove(checkpoint_paths.pop(0))

                else:
                    epochs_without_improvement += 1

                # Early stopping: stop training if no improvement after `patience` epochs
                if epochs_without_improvement >= patience:
                    print(f"Early stopping: No improvement in validation loss for {patience} steps.")
                    return

                # Step the learning rate scheduler based on validation loss
                scheduler.step(avg_val_loss)

            else:
                avg_val_loss = '-'
                avg_train_loss = total_loss_step_window / (global_step % val_interval or 1)

            # Update progress bar
            pbar.set_postfix({
                "train_loss": f"{avg_train_loss:.4f}",
                "avg_val_loss": avg_val_loss if isinstance(avg_val_loss, str) else f"{avg_val_loss:.4f}"
            })

def validate(fabric, model, dataloader):
    fabric_model = model.module if hasattr(model, "module") else model
    fabric_model.eval()

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in dataloader:
            _, loss = model(x, y)
            reduced_loss = fabric.all_reduce(loss.detach(), reduce_op="mean")
            total_loss += reduced_loss.item()
            count += 1

    fabric_model.train()
    return total_loss / count if count > 0 else float('inf')

def main(args):
    # Lightning Fabric Setup
    fabric = L.Fabric(
        accelerator=args.accelerator,
        strategy=args.dist_backend,
        devices=args.gpus,
        precision=args.precision,
    )
    fabric.launch()

    # Dataset and DataLoaders
    train_dataset = TrainDataset(args.corpus_folder, fabric, args.block_size)
    val_dataset = ValDataset(args.corpus_folder, fabric, args.block_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Setup Fabric DataLoaders
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # Model & Optimizer
    model = GPTModel(
        vocab_size=train_dataset.vocab_size,
        block_size=args.block_size,
        n_embed=384,
        n_head=3,
        n_layer=4,
        dropout=0.2,
        device=args.accelerator
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Learning Rate Scheduler: Reduce learning rate on plateau (validation loss)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.6)

    model, optimizer = fabric.setup(model, optimizer)

    # Training Loop
    train(fabric, model, optimizer, scheduler, train_loader, val_loader, max_epochs=args.max_epochs, val_interval=args.val_interval)

if __name__ == "__main__":
    import argparse

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(description="Train GPT Model with Fabric")

    # Fabric / Device
    parser.add_argument('-a', '--accelerator', default="auto", help="Training device")
    parser.add_argument('-g', '--gpus', default=1, type=int, help='Number of devices per node')
    parser.add_argument('-w', '--num_workers', default=4, type=int, help='Data loading workers')
    parser.add_argument('-db', '--dist_backend', default='auto', type=str, help='Distributed backend strategy')

    # Dataset
    parser.add_argument('-cp', '--corpus_folder', required=True, type=str, help='Folder with .txt training files')

    # Training Hyperparameters
    parser.add_argument('-e', '--max_epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('-v', '--val_interval', default=100, type=int, help='Steps between validations')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', default=2e-4, type=float, help='Learning rate')
    parser.add_argument('--block_size', default=128, type=int, help='Block size (seq len for GPT)')
    parser.add_argument('--precision', default='16-mixed', type=str, help='Precision setting')

    args = parser.parse_args()
    main(args)
