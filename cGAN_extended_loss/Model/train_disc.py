import torch
import torch.nn as nn

def train_discriminator(x_data, y_true, BCE_criterion, gen, disc, opt_disc):
    y_fake = gen(x_data)
    D_real = disc(x_data, y_true)
    D_fake = disc(x_data, y_fake.detach())
    D_real_Loss = BCE_criterion(D_real, torch.ones_like(D_real))
    D_fake_Loss = BCE_criterion(D_fake, torch.zeros_like(D_fake))
    D_Loss = (D_real_Loss + D_fake_Loss) / 2

    disc.zero_grad()   #have a question!?
    D_Loss.backward()
    opt_disc.step()
    
    return y_fake, D_Loss