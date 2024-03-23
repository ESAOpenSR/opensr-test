# 

## The opensr-test Dataset

<p align="center">
    <a href="/docs/api.md"><img src="../images/paper_figure01.png" alt="opensr-test"></a>
</p>

The opensr-test dataset was meticulously prepared with the primary aim of maintaining the maximum possible consistency between the `LR` and `HR` pair images. The entire process can be divided into three steps: selection of potential LR-HR pairs, harmonization, and visual inspection, as depicted in Figure above. For the LR-HR pair selection, we utilized Sentinel-2 L2A as the LR image and three different pre-processed HR sources: NAIP, SPOT and Venus. In order to ensure similar atmospheric conditions, we started to discard all the LR-HR pairs that did not count on the same day. Additionally, LR-HR pairs with over 0 % cloud cover in the `LR` image were automatically discarded using a cloud detection algorithm trained on CloudSEN12. The final dataset characteristics are summarized in the table below.


| Dataset | Scale | # Scenes | HRsize | HR Reference |
|----------------|-------|----------|--------|--------------------------------------------------------------------------------------------|
| NAIP        | x4    | 30      | 512    | USDA Farm Production and Conservation - Business Center, Geospatial Enterprise Operations. |
| SPOT     | x4    | 12      | 512    | European Space Agency, 2017, SPOT 1-5 ESA                                                  |
| Mini-SEN2VENµS | x2    | 59      | 512    | Vegetation and Environment monitoring on a New Micro-Satellite (SEN2VENμS).
|

The `opensr-test` dataset API provides a simple interface to download the dataset.  All the dataset are stored in the HuggingFace Datasets Repository https://huggingface.co/csaybar/opensr-test. The following code snippet shows how to download the dataset and load it into memory.

```python
import opensr_test

# Load the spot dataset
lr, hr, landuse, parameters = opensr_test.load("spot").values()


# Load the venus dataset
lr, hr, landuse, parameters = opensr_test.load("venus").values()


# Load the naip dataset
lr, hr, landuse, parameters = opensr_test.load("naip").values()
```

## `opensr_test.load` Function

The `opensr_test.load` function is a key utility in the `opensr-test` package, offering a streamlined solution for downloading and loading datasets. It returns a dictionary containing the following keys:

- `lr`: LR images sourced from Sentinel-2 L2A, representing the base resolution for analysis.
- `hr`: HR images that are harmonized in reference to the `LR` images. These HR images can br from sources like NAIP, SPOT, or VENUS according to the dataset parameter.
- `landuse`: Land use classification images, derived from the ESA Land Cover map, providing contextual information about the geographical area covered by the LR and HR images.
- `parameters`: A datamodel encompassing various fields that guide image processing and analysis. These include:

    - `blur_gaussian_sigma`: Specifies the standard deviation for a Gaussian 2D filter, which can be used to blur the `hr` images. This parameter is fine-tuned comparing the LR and HR images, optimized against the L1 loss metric.

    - `stability_threshold`: A threshold value used to differentiate between stable and unstable pixels. Stable pixels are those that exhibit minimal or no change between the `lr` and `hr` images. This threshold is established through visual inspections performed by three remote sensing experts on each image.
    
    - `correctness_params`: Global parameters delineating the spaces for omission, improvement and hallucinations. These are determined through the expert analysis of three different specialists, ensuring a rigorous and comprehensive evaluation.
    
    - `downsample_method`: The recommended approach for downsampling the `hr` images to a lower resolution.
    
    - `upsample_method`: The suggested technique for upsampling the `lr` images to a higher resolution.
