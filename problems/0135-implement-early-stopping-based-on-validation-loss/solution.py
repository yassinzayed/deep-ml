from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    best_loss = val_losses[0]
    best_epoch = 0
    no_improvement = 0
    for i in range(1, len(val_losses)):
        change = best_loss - val_losses[i]
        if change > min_delta:
            best_loss = val_losses[i]
            best_epoch = i
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= patience:
            return (i, best_epoch)
    return (len(val_losses) - 1, best_epoch)