# FELinterp
Implementations of various FEL Lamp Irradiance models for interpolation and uncertainty propagaion

| Class | Model Description | Reference | 
| ----- | ----------------- | ----------|
| BaseModel | Abstract Base Class for FEL interpolation models | |
| SSBUV     | Shuttle Solar Backscatter Ultraviolet (SSBUV) model that uses a single black body emission expression for the entire wavelength region. The lamp emissivity model is split at _LAMBDA0 | Huang et al., 1998. New procedure for interpolating NIST FEL lamp irradiances |
| SSBUVw0   | Same as above, but with the w0 = _LAMBDA0 as a fitting parameter | |
| NIST | NIST 4th order polynomial-modifed Wien Approximation.  Fit over 3 distint regions (250-350, 350-800, 800-2400 nm) yield 3 separate lamp color temperatures. | Yoon and Gibson, 2011. NIST Measurement Services: Spectral Irradiance Calibrations|

Individual lamp calibration and class-based uncertainty ata from four Optroinc Labs FEL lamps are provided the lamps subdirectory.

