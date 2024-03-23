# 

## The correctness metrics

The correctness metrics evaluate the quality of high-frequency information introduced by the SR model. Three correctness metrics are implemented in `opensr-test`: improvement, omission, and hallucination. Each metric serves a specific purpose:

**Improvement**: Low values represent a good match between the SR and HR images. The equation for calculating improvement is:

$H = d_{im} + d_{om} - 1$

$Improvement = d_{im} + d_{om}*(1 - e^{-\gamma H})$

<p align="center">
  <img src="../images/improvement.png" alt="correctness_metric">
</p>

Where:

- $d_{im}$ is the distance between the SR and HR images.
- $d_{om}$ is the distance between the SR and LR images.

**Omission**: Low values are related to the inability to represent the actual high-frequency information from the landscape and keep similar to the LR image. The equation for calculating improvement is:

$H = d_{im} + d_{om} - 1$

$Omission = d_{om} + d_{im}*(1 - e^{-\gamma H})$

<p align="center">
  <img src="../images/omission.png" alt="correctness_metric">
</p>

**Hallucination**: Low values shows the areas where the SR model has introduced high-frequency information that is not present in the HR image. The equation for calculating hallucination is:

$Hallucination = e^{-\gamma * d_{om} * d_{im}}$

<p align="center">
  <img src="../images/hallucination.png" alt="correctness_metric">
</p>

