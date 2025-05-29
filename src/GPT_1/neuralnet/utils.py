import os
import torch

def set_barrier(fabric):
    fabric.barrier()

def save_best_checkpoint(fabric, model, checkpoint_dir, epoch, val_loss, best_val_loss, epochs_without_improvement, checkpoint_paths):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0

        if fabric.is_global_zero:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}_val_loss_{best_val_loss:.4f}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_paths.append(checkpoint_path)

            if len(checkpoint_paths) > 5:
                os.remove(checkpoint_paths.pop(0))
    else:
        epochs_without_improvement += 1

    return best_val_loss, epochs_without_improvement

def should_stop_early(epochs_without_improvement, patience, fabric):
    if epochs_without_improvement >= patience:
        if fabric.is_global_zero:
            print(f"Early stopping: No improvement in validation loss for {patience} epochs.")
        return True
    return False
