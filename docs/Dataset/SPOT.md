#

# The SPOT product

The SPOT product were obtained from the Worldstrat dataset. The RGBN bands were downsampled to 1.5 m/pixel using the band pan-sharpening method defined in the worldstrat repository, and then to 2.5 m/pixel using bilinear interpolation with the intention to provide a 4x super-resolution factor. 
Due to the irreversible nature of the pansharpening method, which is theoretically applicable primarily to RGB bands, a significant degradation of spectral information occurs in most images within the worldstrat collection. As a result of this degradation, only 12 Regions of Interest (ROIs) have retained sufficient quality.

<p align="center">
  <img src="../images/spot1.png"/>  
</p>