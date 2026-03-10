
def l1_penalty(loss, model, l1_lambda=0.01):
    """
    Fügt L1-Regularisierung zu einem gegebenen Loss hinzu.

    Args:
        loss (torch.Tensor): Der ursprüngliche Loss-Wert.
        model (torch.nn.Module): Das Modell, dessen Parameter reguliert werden sollen.
        l1_lambda (float): Der Regularisierungsfaktor für L1.

    Returns:
        torch.Tensor: Der Loss mit der L1-Regularisierung.
    """

    # Add L1 regularization
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    loss += l1_lambda * l1_norm
    return loss
