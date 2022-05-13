# Machine-Learning-in-MRI
Modified CGAN with extended loss by Fourier for MR image reconstruction.

## Description
Concept of modified CGAN. Deep learning network has additional Fourier term in loss function
![alt text](https://github.com/visualIgorec/Machine-Learning-in-MRI/blob/main/picture/CGAN_scheme.png)

- Network was learned on heart dataset that includes diverse type of heart images with 1.5T and 3T. Link is in Reference below for getting more information about dataset and link for net_param too.
- You can use network for your own task using learned weights parameters as transfer learning in order to simplify learning process.

## Result
- Here you can see prediction and ground truth images
![alt text](https://github.com/visualIgorec/Machine-Learning-in-MRI/blob/main/picture/Снимок.PNG)
![alt text](https://github.com/visualIgorec/Machine-Learning-in-MRI/blob/main/picture/Снимок1.PNG)
![alt text](https://github.com/visualIgorec/Machine-Learning-in-MRI/blob/main/picture/Снимок2.PNG)

## Reference
- Learned net_param: https://drive.google.com/file/d/17fj30xn7XFoKGzMnjumZ7sWdRl4xz4zq/view?usp=sharing
- OCMR dataset: https://arxiv.org/abs/2008.03410v2
- Original Paper CGAN: https://arxiv.org/abs/1411.1784
