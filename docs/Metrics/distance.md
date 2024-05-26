#

<p align="center">
  <img src="../images/distance_metric.png" alt="distance_metric" width="50%">
</p>


## **Distance metrics**

The `opensr-test` package offers a comprehensive suite of nine different distance metrics designed to assess the consistency, synthesis, and accuracy of super-resolution models. These metrics are structured such that a score of zero represents optimal performance, with higher scores indicating decreasing model effectiveness. The distance metrics available in the `opensr-test` package include:

**L1 distance**: The L1 distance is the sum of the absolute differences between the two vectors. It is also known as the Manhattan distance. 

$L1(y, ŷ) = \frac{1}{n} \sum_{i=1}^{n} |y_i - ŷ_i|$

**L2 distance**: The L2 distance is the square root of the sum of the squared differences between the two vectors. It is also known as the Euclidean distance.

$L2(y, ŷ) = \frac{1}{n} \sum_{i=1}^{n} (y_i - ŷ_i)^2$

**Spectral angle distance**: The spectral angle distance is the angle between two vectors. The angle is estimated in degrees. 

$SAD(\vec{y}, \vec{\hat{y}}) = \arccos\left(\frac{\vec{y} \cdot \vec{\hat{y}}}{\|\vec{y}\| \|\vec{\hat{y}}\|}\right)$

**Percentage Bias**: The Percentage Bias (PBIAS) measures the average tendency of the super-resolved values to be larger or smaller than their observed counterparts. This metric help us to understand whether a model is changing the norm of the original reflectance values. The equation for calculating Percentage Bias is:

$PBIAS = \left( \frac{\sum_{i=1}^{n} (O_i - S_i)}{\sum_{i=1}^{n} O_i} \right)$

**Inverted Peak Signal-to-Noise Ratio**: The Inverted Peak Signal-to-Noise Ratio (IPSNR) is the inverse of the Peak Signal-to-Noise Ratio (PSNR). The equation for calculating IPSNR is:

$\text{PSNR} = \text{MAX}_I^2 \cdot 10^{-\frac{\text{MSE}}{10}}
$

Where $\text{MAX}_I$ is set to 1, and $\text{MSE}$ is the Mean Squared Error.


**Kullback-Leibler divergence**: The Kullback-Leibler divergence (KLD) is a measure of how one probability distribution is different from a second, reference probability distribution. The equation for calculating KLD is:

$KL(P || Q) = \sum_{x} P(x) \log\left(\frac{P(x)}{Q(x)}\right)$

**LPIPS**: The Learned Perceptual Image Patch Similarity (LPIPS) metric is a perceptual metric that aims to quantify the perceptual similarity between two images. The LPIPS metric is based on a deep neural network that was trained to predict perceptual similarity scores. The reported metric is the average LPIPS score between the LR and SR images.

$\text{LPIPS} = \sum_{l=1}^{L} w_l \cdot \frac{1}{H_lW_l} \sum_{h=1}^{H_l} \sum_{w=1}^{W_l} \| \phi_l(I_1)_{h,w} - \phi_l(I_2)_{h,w} \|_2^2$

In this equation:
    
- $LPIPS$ is the Learned Perceptual Image Patch Similarity score.
- $L$ denotes the number of layers in a deep neural network used for comparison.
- $w_l$ represents the weight of the $l-th$ layer in the network.
- $\phi_l(l)_{I}$ is the feature map of image $I$ at layer $l$.
- $H_l$ and $W_l$ are the height and width of the feature map at layer $l$, respectively.
- $I_1$ and $I_2$ are the two images being compared.
- The summations across $h$ and $w$ are over the spatial dimensions of the feature maps.

**CLIP**

CLIP measures the distance (L1) in image embedding space, see [CLIP](https://github.com/openai/CLIP) model. Unlike LPIPS, with CLIPscore we can focus mainly in the contextual and semantic integrity of the super-resolved images. We use the [**RemoteCLIP**](https://arxiv.org/pdf/2306.11029) pretrained model. Please cite the following paper if you use this metric:

```bibtex
@article{liu2024remoteclip,
  title={Remoteclip: A vision language foundation model for remote sensing},
  author={Liu, Fan and Chen, Delong and Guan, Zhangqingyun and Zhou, Xiaocong and Zhu, Jiale and Ye, Qiaolin and Fu, Liyong and Zhou, Jun},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```

**MTF**

The Modulation Transfer Function (MTF) is a metric that measures the ability of an imaging system to reproduce the spatial frequencies of an object. The MTF is calculated as the ratio of the SR image to the HR image in the frequency domain:

$MTF = \frac{|\text{FFT}(HR) - \text{FFT}(SR)|}{|\text{FFT}(HR)|}$

Where $\text{FFT}$ is the Fast Fourier Transform after the Sentinel-2 Nyquist frequency.