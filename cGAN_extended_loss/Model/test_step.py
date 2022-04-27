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

def test_step(test_loader, BCE_criterion, Loss, gen, disc, device, common_list_test, alpha):
    
    for index_b, (feature, target) in enumerate(test_loader):
    
        x_data = feature.to(device)
        y_true = target.to(device)
        
        #discriminator check
        y_fake = gen(x_data)
        D_real = disc(x_data, y_true)
        D_fake = disc(x_data, y_fake.detach())
        D_real_Loss = BCE_criterion(D_real, torch.ones_like(D_real))
        D_fake_Loss = BCE_criterion(D_fake, torch.zeros_like(D_fake))
        D_Loss = (D_real_Loss + D_fake_Loss) / 2
        
        #generator check
        D_fake = disc(x_data, y_fake)
        G_fake_Loss = BCE_criterion(D_fake, torch.ones_like(D_fake))
        loss = common_list_test[0] * Loss(y_fake, y_true)
        G_Loss = G_fake_Loss + loss + alpha*Loss_Fourier(y_fake, y_true)
        
        common_list_test[-2].append(G_Loss.item())
        common_list_test[-1].append(D_Loss.item())
    
    g_mean_loss = sum(common_list_test[-2])/len(common_list_test[-2])
    d_mean_loss = sum(common_list_test[-1])/len(common_list_test[-1])
    
    common_list_test[1].append(g_mean_loss)
    common_list_test[2].append(d_mean_loss)
    
    return y_fake, y_true, g_mean_loss, d_mean_loss