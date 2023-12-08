# 

## A framework for remote sensing image SR evaluation

Assessing the quality of a SR model is a challenging task. Drawing inspiration from Wald et al. 1997, who outlined key properties for ideal pansharpening algorithms, we discuss the three properties that a SR model must satisfy to be considered a *good* SR model. These properties are: consistency, synthesis, and correctness.

<p align="center">
  <img src="../images/properties.png" alt="properties" width="65%">
</p>

### First property (Consistency)

Any SR image when degraded to the original LR spatial resolution SR, must maintain consistent reflectance values and spatial alignment with their LR counterparts. Testing this property can be challenging due that the degradation model to generate LR from SR typically is unknown. Assuming that the high-frequency characteristics between SR and HR are similar, the degradation model learned from HR and LR could be employed to degrade the SR image to the LR resolution (SRup).

### Second property (Synthesis)

Any SR image must improve the Ground Resolved Distance (GRD), which is the measure of the most minor distinguishable feature that can be detected and resolved in an image. Ideally, the frequencies of the SR image should be as identical as possible to those of the corresponding sensor with a higher resolution. Assuming that the shared frequencies between the LR and SR images remain consistent, a downsampled LR image, LRdown, can be employed as a reference. Then, the GRD can be estimated as the distance between the LRdown and SR images. The larger the distance, the better the SR model accomplished the second property.

### Third property (Correctness)

Any SR model must avoid hallucinations. Hallucinations are high-frequency information that is not present in the HR image. To test this property, we propose to compare the SR image with the HR and LR images. The distance between the SR and HR images (sr_to_hr) should be smaller than the distance between the SR and LR images (sr_to_lr). Both distances are normalized by the distance between the LR and HR images (lr_to_hr). More information about this property can be found in the [correctness metrics](../Metrics/correctness.md) section.