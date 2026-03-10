
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
    
    if l1_lambda == 0.0:
        return loss
     
    l1_norm = 0.0
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            l1_norm += param.abs().sum()

    return loss + l1_lambda * l1_norm
