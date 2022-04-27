import matplotlib.pyplot as plt

def visual(y_true, y_valid_fake):
    
    #Visualization
    img_test = y_valid_fake.cpu().detach()[0][0]
    img_target = y_true.cpu().detach()[0][0]

    fig = plt.figure(figsize=(10,10))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img_test, cmap='gray')
    plt.title("Valid image")

    fig.add_subplot(1, 2, 2)
    plt.imshow(img_target, cmap='gray')
    plt.title("Target image")
    plt.show()