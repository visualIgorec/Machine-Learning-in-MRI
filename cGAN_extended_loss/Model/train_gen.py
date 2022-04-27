import torch
import torch.nn as nn

def Loss_Fourier(input_prediction, input_target):
    output_target = torch.torch.fft.fftn(input_target, dim=(2,3))
    output_target = torch.fft.fftshift(output_target)

    output_prediction = torch.torch.fft.fftn(input_prediction, dim=(2,3))
    output_prediction = torch.fft.fftshift(output_prediction)
    
    difference = torch.abs(output_target - output_prediction)
    output_loss = torch.sum(difference) / torch.numel(input_prediction)
    return output_loss

def train_generator(x_data, y_true, y_fake, BCE_criterion, gen, disc, Loss, opt_gen, common_list, alpha):
    
    D_fake = disc(x_data, y_fake)
    G_fake_Loss = BCE_criterion(D_fake, torch.ones_like(D_fake))
    loss = common_list[0] * Loss(y_fake, y_true)
    G_Loss = G_fake_Loss + loss + alpha*Loss_Fourier(y_fake, y_true) #CHANGED!

    gen.zero_grad()  #have a question!? ----opt_gen
    G_Loss.backward()
    opt_gen.step()
    
    return G_Loss