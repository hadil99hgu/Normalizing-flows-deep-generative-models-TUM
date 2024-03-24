def likelihood(X_train, model, device):
    ##########################################################
    import torch
    X_train = X_train.to(device)
    log_prob = model.log_prob(X_train)
    loss = -torch.mean(log_prob)
    ##########################################################

    return loss
