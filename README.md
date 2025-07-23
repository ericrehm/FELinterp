# FELinterp
Implementations of various FEL Lamp Irradiance models for interpolation and uncertainty propagation.

| Class | Model Description | Reference | 
| ----- | ----------------- | ----------|
| BaseModel | Abstract Base Class for FEL interpolation models | |
| SSBUV     | Shuttle Solar Backscatter Ultraviolet (SSBUV) model that uses a single black body emission expression for the entire wavelength region. The lamp emissivity model is split at _LAMBDA0.  Default is _LAMBDA0 = 450 nm | Huang et al., 1998. [New procedure for interpolating NIST FEL lamp irradiances](https://www.gml.noaa.gov/grad/neubrew/docs/publications/Huang_interpolatiopn.pdf) |
| SSBUVw0   | Same as above, but with the w0 = _LAMBDA0 as a fitting parameter | |
| NIST | NIST 4th order polynomial-modifed Wien Approximation.  One must fit over 3 distinct regions (250-350, 350-800, 800-2400 nm), yielding 3 separate lamp color temperatures.  The currentimplementation fits one user-specifiable region. | Yoon and Gibson, 2011. [NIST Measurement Services: Spectral Irradiance Calibrations](https://doi.org/10.6028/NIST.SP.250-89)|
| WhiteSpline | Demonstrates a method for analysing the uncertainty propagation of interpolating equations that exploits the linear dependence of the interpolations on the input samples of the interpolated quantity. The technique is based on the linear expansion of the interpolating equation as a sum of interpolating functions each multiplied by a corresponding yi value. The interpolating functions are therefore the sensitivity
coefficients for the uncertainties in the yi values. In this case, a piecewise-cubic spline interpolation is used, so the interpolated values are identical to the input irradiance data at the input wavelengths and C2 continuity is guaranteed.| White, 2017. [Propagation of Uncertainty and Comparison of Interpolation Schemes](https://doi.org/10.1007/s10765-016-2174-6)|


Individual lamp calibration and class-based uncertainty data from four Optroinc Labs FEL lamps are provided the lamps subdirectory.
