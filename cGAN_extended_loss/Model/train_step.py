
def train_step(train_discriminator, train_generator, train_loader, BCE_criterion, Loss, gen, disc, opt_disc, opt_gen, device, common_list, alpha):
    
    for index_b, (feature, target) in enumerate(train_loader):
    
        x_data = feature.to(device)
        y_true = target.to(device)
        
        #train discriminator:
        y_fake, D_Loss = train_discriminator(x_data, y_true, BCE_criterion, gen, disc, opt_disc)
        
        #train generator:
        G_Loss = train_generator(x_data, y_true, y_fake, BCE_criterion, gen, disc, Loss, opt_gen, common_list, alpha)
        
        common_list[3].append(G_Loss.item())
        common_list[4].append(D_Loss.item())
    
    g_mean_loss = sum(common_list[3])/len(common_list[3])
    d_mean_loss = sum(common_list[4])/len(common_list[4])
    
    common_list[1].append(g_mean_loss)
    common_list[2].append(d_mean_loss)
    
    return y_fake, y_true, g_mean_loss, d_mean_loss