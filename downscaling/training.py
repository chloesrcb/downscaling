
import numpy as np
import torch
import copy
from torch.utils.data import DataLoader, TensorDataset

from downscaling.models import EGPDNNOnlyInputs
from downscaling.egpd_torch import egpd_left_censored_nll_torch


def egpd_left_censored_nll_loss(y_true, y_pred, c: float = 0.22):
    return egpd_left_censored_nll_torch(
        y=y_true,
        sigma=y_pred[..., 0],
        kappa=y_pred[..., 1],
        xi=y_pred[..., 2],
        censor_threshold=c,
    )


def train_egpd_nn_only(
    model,
    X_s,
    X_k,
    Y_train,
    offset=None,
    X_s_valid=None,
    X_k_valid=None,
    offset_valid=None,
    Y_valid=None,
    n_epochs=100,
    batch_size=64,
    lr=1e-3,
    weight_decay=0.0,
    clipnorm=1.0,
    seed=1,
    device=None,
    dtype=torch.float32,
    early_stopping=True,
    patience=25,
    min_delta=0.0,
    warmup_epochs=20,
    restore_best=True,
    censor_threshold=0.22,
    kappa_max_nn=1.5,
    lambda_kappa=0.05,
    ):
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine device
    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(dev)

    # Function to convert inputs to tensors or return None
    def to_tensor_or_none(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.to(device=dev, dtype=dtype)
        return torch.as_tensor(x, dtype=dtype, device=dev)

    # Convert all inputs to tensors
    Xs = to_tensor_or_none(X_s)
    Xk = to_tensor_or_none(X_k)
    Off = to_tensor_or_none(offset)
    Yt = to_tensor_or_none(Y_train)

    Xs_valid_t = to_tensor_or_none(X_s_valid)
    Xk_valid_t = to_tensor_or_none(X_k_valid)
    Off_valid_t = to_tensor_or_none(offset_valid)
    Yv = to_tensor_or_none(Y_valid)

    # Create DataLoader for training data
    n_train = Yt.shape[0]
    idx_all = torch.arange(n_train, device=dev)

    train_loader = DataLoader(
        TensorDataset(idx_all),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    # Initialize optimizer: Adam
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Inits
    train_epoch_losses = []
    val_epoch_losses = []

    best_score = float("inf")
    best_state = None
    best_epoch = None
    bad_epochs = 0
    stopped_epoch = n_epochs

    # function to subset tensors or return None
    def subset_or_none(x, idx):
        if x is None:
            return None
        return x[idx]

    # function to create model inputs
    def make_inputs(Xs_part, Xk_part, offset_part):
        return EGPDNNOnlyInputs(
            X_s=Xs_part,
            X_k=Xk_part,
            offset=offset_part,
        )

    # Training loop
    for ep in range(1, n_epochs + 1):
        model.train() # Set model to training mode
        batch_losses = [] # List to store batch losses for each epoch

        # Iterate over batches
        for (idx_batch,) in train_loader:
            # for each batch, subset the inputs
            Xs_b = subset_or_none(Xs, idx_batch)
            Xk_b = subset_or_none(Xk, idx_batch)
            Off_b = subset_or_none(Off, idx_batch) # Offset
            Y_b = Yt[idx_batch] # Target variable for the batch
            
            # Forward pass
            pred_b = model(make_inputs(Xs_b, Xk_b, Off_b))
            # Compute loss
            loss_b = egpd_left_censored_nll_loss(
                y_true=Y_b,
                y_pred=pred_b,
                c=censor_threshold,
            )

            # kappa penalty
            pred_kappa = pred_b[..., 1]

            kappa_penalty = torch.mean(
                torch.relu(pred_kappa - kappa_max_nn) ** 2
            )

            loss_b = loss_b + lambda_kappa * kappa_penalty
            
            # Backward pass and optimization step
            opt.zero_grad()
            loss_b.backward()

            # Gradient clipping
            if clipnorm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)

            # Update parameters
            opt.step()
            # Store batch loss
            batch_losses.append(float(loss_b.detach().cpu()))

        # Compute average training loss for the epoch
        train_loss = float(np.mean(batch_losses))
        train_epoch_losses.append(train_loss)

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            if Yv is not None:
                # Compute validation loss, if validation data is provided
                pred_val = model(make_inputs(Xs_valid_t, Xk_valid_t, Off_valid_t))
                val_loss = float(
                    egpd_left_censored_nll_loss(
                        y_true=Yv,
                        y_pred=pred_val,
                        c=censor_threshold,
                    ).detach().cpu()
                )
                val_epoch_losses.append(val_loss)
                score = val_loss
            else:
                # If no validation data, use training loss for early stopping
                score = train_loss

        # Determine if we should use early stopping
        use_early_stopping = early_stopping and (ep > warmup_epochs)
        
        # Check for improvement and update best score/state
        improved = (best_score - score) > min_delta
        if ep <= warmup_epochs:
            if improved:
                best_score = score
                best_epoch = ep
                best_state = copy.deepcopy(model.state_dict())
            continue
        if improved:
            best_score = score
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if use_early_stopping and (Yv is not None) and (bad_epochs >= patience):
            stopped_epoch = ep
            break

    # Restore best model state if requested
    if restore_best and (best_state is not None):
        model.load_state_dict(best_state)

    # Final evaluation on training (and validation if available)
    model.eval()
    with torch.no_grad():
        pred_train = model(make_inputs(Xs, Xk, Off))
        final_train = float(
            egpd_left_censored_nll_loss(
                y_true=Yt,
                y_pred=pred_train,
                c=censor_threshold,
            ).detach().cpu()
        )

        out = {
            "model": model,
            "train_nll": final_train,
            "history": {
                "train_epoch": train_epoch_losses,
                "val_epoch": val_epoch_losses,
            },
            "best_score": float(best_score),
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "stopped_epoch": int(stopped_epoch),
        }

        if Yv is not None:
            pred_val = model(make_inputs(Xs_valid_t, Xk_valid_t, Off_valid_t))
            final_val = float(
                egpd_left_censored_nll_loss(
                    y_true=Yv,
                    y_pred=pred_val,
                    c=censor_threshold,
                ).detach().cpu()
            )
            out["val_nll"] = final_val

    return out
