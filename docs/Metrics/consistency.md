## Consistency metrics

The consistency metrics are used to evaluate the ability of the super-resolution model to preserve the spectral and spatial information of the LR image. The `opensr-test` package provides three different metrics to evaluate the consistency of the super-resolution model: reflectance, spectral, and spatial. See the [result attributes](../API/model_parameters.md#result-attributes) for more information about how to retrieve the outputs.

- The **reflectance metrics** are employed to assess the impact of the SR model on the norm of reflectance values. The distance metric supported are: l1, l2, pbias, and kl.

- The **spectral metrics** are used to evaluate if the SR model is preserving the spectral profiles of the images. The only distance metric supported are: sad.

- The **spatial metrics** are used to evaluate the spatial alignment and structural integrity of the SR image compared to the LR image. The only distance metric supported are: ligthglue+superpoint and lightglue+disk.


**L1 distance**: The L1 distance is the sum of the absolute differences between the two vectors. It is also known as the Manhattan distance. 

$L1(y, ŷ) = \frac{1}{n} \sum_{i=1}^{n} |y_i - ŷ_i|$

**L2 distance**: The L2 distance is the square root of the sum of the squared differences between the two vectors. It is also known as the Euclidean distance.

$L2(y, ŷ) = \frac{1}{n} \sum_{i=1}^{n} (y_i - ŷ_i)^2$

**Spectral angle distance**: The spectral angle distance is the angle between two vectors. This metric is used to measure the spectral consistency.

$SAD(\vec{y}, \vec{\hat{y}}) = \arccos\left(\frac{\vec{y} \cdot \vec{\hat{y}}}{\|\vec{y}\| \|\vec{\hat{y}}\|}\right)$

**Percentage Bias**: The Percentage Bias (PBIAS) measures the average tendency of the super-resolved values to be larger or smaller than their observed counterparts. This metric help us to understand whether a model is changing the norm of the original reflectance values. The equation for calculating Percentage Bias is:

$PBIAS = \left( \frac{\sum_{i=1}^{n} (O_i - S_i)}{\sum_{i=1}^{n} O_i} \right)$

**Inverted Peak Signal-to-Noise Ratio**: The Inverted Peak Signal-to-Noise Ratio (IPSNR) is the inverse of the Peak Signal-to-Noise Ratio (PSNR). This metric is used to measure the high-frequency information. The equation for calculating IPSNR is:

$IPSNR = \frac{1}{PSNR}$

**Kullback-Leibler divergence**: The Kullback-Leibler divergence (KLD) is a measure of how one probability distribution is different from a second, reference probability distribution. This metric is used to measure the high-frequency information. The equation for calculating KLD is:

$KL(P || Q) = \sum_{x} P(x) \log\left(\frac{P(x)}{Q(x)}\right)$