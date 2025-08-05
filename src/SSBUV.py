from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import numdifftools as nd
from punpy.mc.mc_propagation import MCPropagation
import plotly.graph_objects as go
from BaseModel import BaseModel


@dataclass
class SSBUV(BaseModel):
    ''' 
    SSBUV model, Huang et al. 1998, "New procedure for interpolating NIST FEL lamp irradiances"
    '''

    # SSBUV-specific attributes
    log_flux_data : np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    log_flux_unc  : np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    _LAMBDA0 : float = 450
    # _LAMBDA0 : float = 496.5  # F1711
    # _LAMBDA0 : float = 495.5   # F1738

    # SSBUV methods (some override BaseModel methods)
    def __post_init__(self) :
        ''' Initialize the subclass by transforming data, doing the curve_fit plus anything in BaseModel '''
        super().__post_init__() 
    
        self.names = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        self.log_flux_data = self.transform(self.wl_data, self.irr_data)
        self.log_flux_unc  = self.unc_data_abs_k1/self.irr_data
    
        # Initial guesses for parameters: C0,...,C6
        self.p0 = np.array([10,-1000,-1e-4,-1e3,10,-0.01,1])  # SSBUV

        # Fit using weighted, non-linear least squares. (ML estimator since input errors are provided.)
        (self.params, self.pcov, _, _, _) = curve_fit(        # type: ignore
            self._model_transformed_internal,
            self.wl_data,
            self.log_flux_data,
            sigma=self.log_flux_unc,
            absolute_sigma=True,
            p0 = self.p0,
            maxfev = 10000,
            method = 'lm',
            full_output=True
        )
        self.perr = np.sqrt(np.diag(self.pcov))   # stdev of model params
    
    def transform(self, wl: np.ndarray, irr: np.ndarray) -> np.ndarray:
        '''Untransform from irradiance to log flux'''
        return np.log(wl**5 * irr)
    
    def untransform(self, wl: np.ndarray, log_flux: np.ndarray) -> np.ndarray:
        '''Untransform from log flux to irradiance'''
        return np.exp(log_flux) * (wl**-5)
    
    def model_transformed(self, wavelength):
        '''Compute model results transformed to irradiance'''
        return self._model_transformed_internal(wavelength, *self.params)
    
    def _model_transformed_internal(self, wavelength, *params):
        '''Compute model results in log flux, passing params (used by curve_fit during fitting)'''
        # Retrieve model parameters
        C0, C1, C2, C3, C4, C5, C6 = params

        # Pre-allocate model result F
        F = np.zeros_like(wavelength, dtype=float)

        # Divide SSBUV model at LAMBDA0, per Huang et al. 1998. However,
        # the paper chooses LAMBDA0.  
        # I found the optmization somewhat senstive to the choice of LAMBDA0
        # but was not satisified with the results when LAMBDA0 used as an 
        # optimizationparameter (using again one more degree of freedom).  So 
        # I tuned LAMBDA0 by hand until I saw little change in the residuals.
        mask_lo = wavelength < self._LAMBDA0
        mask_hi = ~mask_lo

        # For wavelengths < LAMBDA0    (Eqn 4)
        F[mask_lo] = (
            C0
            + C1 / wavelength[mask_lo]
            + C2 * wavelength[mask_lo]
            + C3 * np.abs((wavelength[mask_lo] - self._LAMBDA0)/500) ** C4
        )
        # For wavelengths >= LAMBDA0.  (Eqn 5)
        F[mask_hi] = (
            C0
            + C1 / wavelength[mask_hi]
            + C2 * wavelength[mask_hi]
            + C5 * np.abs((wavelength[mask_hi] - self._LAMBDA0)/500) ** C6
        )
        return F
    
    def model(self, wavelength : np.ndarray) :
        '''Compute model results in irradiance'''
        log_flux = self.model_transformed(wavelength.astype(float))
        irr = self.untransform(wavelength, log_flux)
        return irr

    def model_unc(self, wavelength, nsamples, method = 'covariance', doPlot = False) -> pd.DataFrame:
        match method:
            case 'MCPropagation':
                return self.model_unc_mc(wavelength, nsamples)
            case 'bootstrap':
                return self.model_unc_bootstrap(wavelength, nsamples, doPlot)
            case 'covariance':
                return self.model_unc_covariance(wavelength, nsamples)
            case _:
                raise ValueError(f"Unknown method: {method}. Use 'bootstrap', 'covariance, or 'MCPropagation'.")
            

    def model_unc_mc(self, wavelength, nsamples) -> pd.DataFrame:
        '''
        Estimate uncertainty first interpolating original uncertainty data (dubious)
        transformed to log flux space, then resampling log flux with interpolated
        uncertainties.  Not really an MC approach, as the model is not exercised,
        so this uncertainty estimation method has little value.
        '''
        # Set up punpy MC uncert propagation
        mc = MCPropagation(steps=nsamples)

        # Vectorized interpolation wavelengths
        log_flux_pred = self.model_transformed(wavelength)
        log_flux_unc = np.interp(wavelength, self.wl_data, self.log_flux_unc)  # Cheat #1: Linear interp of transformed uncertainties

        # Identity matrix for uncorrelated samples
        corr_matrix = np.eye(len(wavelength))    # Cheat #2: Fit parameters, hence irraidiances may be correlated

        # Now generate MC samples of log_flux data 
        # using interpolated uncertainties and corr_matrix
        samples = mc.generate_MC_sample(
            x=log_flux_pred,
            u_x=log_flux_unc,
            corr_x=corr_matrix
        )

        # Ensure all samples are NumPy arrays first
        samples = [np.asarray(s, dtype=np.float64) for s in samples]

        # Now safely stack them
        samples = np.stack(samples, axis=1).T 
        print(f'(n_MC_samples,) = {samples[0].shape}')  # should be (n_MC_samples,)
        print(f'(n_wavelengths, n_MC_samples) = {samples.shape}')     # should be (n_wavelengths, n_MC_samples)

        # Compute irradiance samples (inverse transform)
        irr_samples = self.untransform(wavelength[:, None], samples)

        # Compute irradiance statistics
        irradiance_pred = np.nanmean(irr_samples, axis=1)
        irradiance_std  = np.nanstd(irr_samples, axis=1)
        irradiance_p025 = np.percentile(irr_samples, 2.5, axis=1)
        irradiance_p975 = np.percentile(irr_samples, 97.5, axis=1)

        df_interp = pd.DataFrame({
            'wavelength': wavelength,
            'irradiance': irradiance_pred,
            'uncertainty': irradiance_std,
            'low_CI': irradiance_p025,
            'high_CI': irradiance_p975
        })

        return df_interp
    
    def model_unc_covariance(self, wavelength, _) -> pd.DataFrame:
        '''
        Estimate uncertainty in interpolated points using curve_fit (input parameter) covariance matrix Cp (7 x 7)
        The Jacobian Jp matrix [∂E_i/∂P_j] (E = irradiance, P = fitting params) is calculated. (n_wavelengths x 7)
        Then the confidence intervals are the square root of the diagonal elements of the output covariance matrix Cy:
            Cy = Jp x Cp x Jp'  (n_wavelengths x 1)
        Computationally, for each wavelength index i, compute Cy(i) = Jp(i, :) x Cp x Jp(i,:)' = (1 x 7) X (7 x 7) X (7 x 1)
        See https://stackoverflow.com/questions/77956010/how-to-estimate-error-propagation-on-regressed-function-from-covariance-matrix-u
        and Arras 1998, "An Introduction To Error Propagation: Derivation, Meaning and Examples of Equation Cy = FCF' 
        https://infoscience.epfl.ch/server/api/core/bitstreams/20ca2fc1-b9b7-4316-a30a-938cef8b00a8/content 

        Note:  Since pcov was computed in log flux space, so is output variance() and the resulting ci.
        To compute the final confidence interval in irradiance space, do ci = untransform(logflux + ci_logflux) - yhat
        Since it's a nonlinear transformation, we should also compute lower ci = yhat - untransform(logflux - ci_logflux), 
        but the upper and lower ci's are very close: within +/- 0.01% of total uncertainty and generally the upper ci >= lower ci.
        '''
        def variance(x, Cp, *p):
            
            def proxy(q):
                log_flux = self._model_transformed_internal(x, *q)
                # irr = self.untransform(x, log_flux)
                # return irr
                return log_flux
            
            def projection(J):
                return J.T @ Cp @ J
                # return J @ Cp @ J.T
            
            Jp = nd.Gradient(proxy)(*p)
            Cy = np.apply_along_axis(projection, 1, Jp)   # Compute output covar major diag
            
            return Cy
        
        # Compute output covariance and k=1 uncertainty in logflux space
        alpha = 1 - 0.6827 # Corresponds k = 1
        # alpha = 1 - 0.95 # Corresponds k = 2
        z = stats.norm.ppf(1 - alpha / 2.)
        Cy = variance(wavelength, self.pcov, self.params) 
        sy = np.sqrt(Cy)
        ci_logflux = z * sy

        # Final transformation back to irradiance space
        yhat = self.model(wavelength)
        logflux = self.transform(wavelength, yhat)
        yhatpci = self.untransform(wavelength,  logflux + ci_logflux)
        ci = yhatpci - yhat

        df_interp = pd.DataFrame({
            'wavelength': wavelength,
            'irradiance': yhat,
            'uncertainty': ci,
        })

        return df_interp



    def model_unc_bootstrap(self, wavelength, nsamples, doPlot) -> pd.DataFrame:
        '''
        Estimate uncertainty in interpolated points by bootstrapping the transformed (!)
        mdoel input data. The curve fit is repeated for each bootstrap sample, based 
        on the original lamp and transformed k=2 uncertainty data.  The mean of the 
        interpolated irradiance is returned, along with the standard deviation of the
        interpolated irradiance samples.
        This may be a more robust approach than MCPropagation method above.
        '''
        # Create a new random number generator for reproducibility
        rng = np.random.default_rng(42)
        log_flux_bootstrap = []

        if doPlot:
            fig = go.Figure()

        # Run bootstrap
        print(f'Bootstrapping {nsamples} samples...')
        for _ in range(nsamples):
            # Add random perturbation to the log flux data (more stable results)
            perturbation = rng.normal(0, self.log_flux_unc)
            log_flux_data_perturbed = self.log_flux_data + perturbation

            # Alternatively, perturb the irradiance data
            # perturbation = rng.normal(0, self.unc_data_abs_k1)
            # irr_data_perturbed = self.irr_data + perturbation
            # log_flux_data_perturbed = self.transform(self.wl_data, irr_data_perturbed)

            # Fit the model to the perturbed data
            try:
                (popt_i, _, _, _, _) = curve_fit(                    # type: ignore
                            self._model_transformed_internal,
                            self.wl_data,
                            log_flux_data_perturbed,
                            p0 = self.p0,
                            maxfev = 10000,
                            method = 'lm',
                            full_output=True
                )

                log_flux = self._model_transformed_internal(wavelength, *popt_i)
                log_flux_bootstrap.append(log_flux)

                if doPlot:
                    pred = self.untransform(wavelength, log_flux)  # shape: (len(wavelength),)
                    fig = fig.add_trace(go.Scatter(
                        x=wavelength, y=pred, mode='lines',
                        name=f'Bootstrap Sample {_+1}', line=dict(color='grey', width=0.5)
                    ) )
            except RuntimeError:
                continue

        if doPlot:
            fig.show('browser')

        # perturbed log_flux data from nsamples model fits and evalulation at interpolated wavelengths
        log_flux_bootstrap = np.array(log_flux_bootstrap)  # shape: (n_samples, len(wwavlength))

        # Inverse transform to irradiance
        irr_samples = self.untransform(wavelength, log_flux_bootstrap)   # shape: (n_samples, len(w_interp))

        # Compute statistics
        irr_mean = np.nanmean(irr_samples, axis=0)
        irr_std = np.nanstd(irr_samples, axis=0)

        # Package interpolated results
        df_interp = pd.DataFrame({
            'wavelength': wavelength,
            'irradiance': irr_mean,
            'uncertainty': irr_std
        })

        return df_interp


@dataclass
class SSBUVw0(SSBUV):
    ''' 
    SSBUVw0 model, Huang et al. 1998, "A New Model for the Spectral Irradiance of a Standard Lamp"

    Same as SSBUV model above, but with _LAMBDA0 = w0 as an optimization parameter.  
    So, inherits from SSBUV, but overrides just __post_init__() and _model_transformed_internal().
    '''

    # SSBUV-specific attributes
    log_flux_data : np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    log_flux_unc  : np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    # _LAMBDA0 : float = 450.5
    _LAMBDA0 : float = 496.5  # F1711
    # _LAMBDA0 : float = 498.4  # F1738

    # SSBUV methods (some override BaseModel methods)
    def __post_init__(self) :
        ''' Initialize the subclass by transforming data, doing the curve_fit plus anything in BaseModel '''
        BaseModel.__post_init__(self)
    
        self.names = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'w0']
        self.log_flux_data = self.transform(self.wl_data, self.irr_data)
        self.log_flux_unc  = self.unc_data_abs_k1/self.irr_data
    
        # Initial guesses for parameters: C0,...,C6
        self.p0 = np.array([10,-1000,-1e-4,-1e3,10,-0.01,1, 450.0])  # SSBUV

        # Fit using weighted, non-linear least squares
        (self.params, self.pcov, _, _, _) = curve_fit(        # type: ignore
            self._model_transformed_internal,
            self.wl_data,
            self.log_flux_data,
            sigma=self.log_flux_unc,
            absolute_sigma=True,
            p0 = self.p0,
            maxfev = 10000,
            method = 'lm',
            full_output=True
        )
        self.perr = np.sqrt(np.diag(self.pcov))   # stdev of model params
        
    def _model_transformed_internal(self, wavelength, *params):
        '''Compute model results in log flux, passing params (used by curve_fit during fitting)'''
        # Retrieve model parameters
        C0, C1, C2, C3, C4, C5, C6, w0 = params
        self._LAMBDA0 = w0  # Update LAMBDA0 based on fit parameter

        # Pre-allocate model result F
        F = np.zeros_like(wavelength, dtype=float)

        # Divide SSBUV model at LAMBDA0, per Huang et al. 1998.
        # T he paper chooses LAMBDA0 = 450 nm
        # I found the optmization VERY senstive to the choice of LAMBDA0
        # but was not satisified with the results when LAMBDA0 used as an 
        # optimizationparameter (using again one more degree of freedom).  
        mask_lo = wavelength < self._LAMBDA0
        mask_hi = ~mask_lo

        # For wavelengths < LAMBDA0    (Eqn 4)
        F[mask_lo] = (
            C0
            + C1 / wavelength[mask_lo]
            + C2 * wavelength[mask_lo]
            + C3 * np.abs((wavelength[mask_lo] - self._LAMBDA0)/500) ** C4
        )
        # For wavelengths >= LAMBDA0.  (Eqn 5)
        F[mask_hi] = (
            C0
            + C1 / wavelength[mask_hi]
            + C2 * wavelength[mask_hi]
            + C5 * np.abs((wavelength[mask_hi] - self._LAMBDA0)/500) ** C6
        )
        return F

#--------------------


def GrayBody(wavelength, a, b, C):
    """
    Calculates irradiance based on a model of a gray body.
    Coefficient inputs are generally determined from curve-fitting.
    The a and b coefficients are determined first from curve-fitting, then used with the gray body model to determine C, also through curve-fitting

    Gray body model assumed:
        E_lambda = (C_0 + C_1 * lambda + C_2 * lambda**2 + ... + C_n * lambda**n) * lambda**-5 * e^(a + b / lambda)

    ----------------------------------------------------------------------------
    Notes
        Authored by: Michael Braine, Physicist, NIST Gaithersburg
        EMAIL: michael.braine@nist.gov
        October 2022

    ----------------------------------------------------------------------------
    References
        "The 1973 NBS Scale of Spectral Irradiance"

    ----------------------------------------------------------------------------
    Inputs
        wavelength  - wavelength or array of wavelengths. Expected units: nm
        a           - coefficient in gray body model, related to gray body emissivity with e**a. Needs determined independently by fitting data (see E_planckian function)
        b           - coefficient in gray body model, related to reciprocal of temperature distribution. Needs determined independently by fitting data (see E_planckian function)
        C           - nx1 array containing gray body coefficients C_0, C_1, ... , C_n

    ----------------------------------------------------------------------------
    Returns
        Irradiance with units W/cm**-3
    """
    return __GrayBody__(a, b)(wavelength, *C)

def __GrayBody__(a, b):
    """
    Gray body model used in curve-fitting. Written to return another function to allow arbitrary number of coefficients in curve-fitting and while enabling substitution of a and b coefficients.
    Usage in this form to yield irradiance: GrayBody_model(a, b)(wavelength, C1, C2, C3, ...)
    **Intended use is curve-fitting to determine C coefficients. See and use the function 'GrayBody' to calculate irradiance from wavelength and coefficients**

    Gray body model assumed:
        E_lambda = (C_0 + C_1 * lambda + C_2 * lambda**2 + ... + C_n * lambda**n) * lambda**-5 * e^(a + b / lambda)

    ----------------------------------------------------------------------------
    Notes
        Authored by: Michael Braine, Physicist, NIST Gaithersburg
        EMAIL: michael.braine@nist.gov
        October 2022

    ----------------------------------------------------------------------------
    References
        NIST TN 594-13 "The 1973 NBS Scale of Spectral Irradiance"

    ----------------------------------------------------------------------------
    Inputs
        a - coefficient in gray body model, related to gray body emissivity with e**a. Needs determined independently by fitting data (see E_planckian function)
        b - coefficient in gray body model, related to reciprocal of temperature distribution. Needs determined independently by fitting data (see E_planckian function)

    ----------------------------------------------------------------------------
    Returns
        GrayBody - function with independent variable (wavelength) and coefficients (C) as inputs
    """
    def GrayBody_model(wavelength, *C):
        """
        Gray body model used in curve-fitting. Written to return another function to allow arbitrary number of coefficients in curve-fitting and while enabling substitution of a and b coefficients.
        Usage in this form to yield irradiance: GrayBody_model(a, b)(wavelength, C1, C2, C3, ...)
        **Intended use is curve-fitting to determine C coefficients. See and use the function 'GrayBody' to calculate irradiance from wavelength and coefficients**

        Gray body model assumed:
            E_lambda = exp(C_2 * lambda + C3 * |(wavelength - 450)/500|**C4, for wavelength < 450
            E_lambda = exp(C_2 * lambda + C5 * |(wavelength - 450)/500|**C6, for wavelength >= 450

        ----------------------------------------------------------------------------
        Notes
            Authored by: Eric Rehm, Sea-Bird Scientifi
            EMAIL: erehm@seabird.com
            August 2025

        ----------------------------------------------------------------------------
        References
            Huang et al. 1998

        ----------------------------------------------------------------------------
        Inputs
            wavelength  - wavelength or array of wavelengths. Expected units: nm
            C           - nx1 array containing gray body coefficients C_2, C_3, ... , C_6

        ----------------------------------------------------------------------------
        Returns
            Irradiance with units W/cm**-3
        """

        # Pre-allocate model result F
        F = np.zeros_like(wavelength, dtype=float)

        # Divide SSBUV model at LAMBDA0, per Huang et al. 1998.
        # The paper chooses LAMBDA0 = 450.  
        # ToDo: I'll fix this with a class variable later.
        mask_lo = wavelength < _LAMBDA0
        mask_hi = ~mask_lo

        C2, C3, C4, C5, C6 = C

        # For wavelengths < LAMBDA0    (Eqn 4)
        F[mask_lo] = np.exp(
            + C2 * wavelength[mask_lo]
            + C3 * np.abs((wavelength[mask_lo] - _LAMBDA0)/500) ** C4
        )

        # For wavelengths >= LAMBDA0.  (Eqn 5)
        F[mask_hi] = np.exp(
            + C2 * wavelength[mask_hi]
            + C5 * np.abs((wavelength[mask_hi] - _LAMBDA0)/500) ** C6
        )
        
        return F*np.array(wavelength)**-5*math.e**(a + b/np.array(wavelength))
    return GrayBody_model

import math
def E_planckian(wavelength, a, b):
    """
    Model to determine coefficients in Gray Body model.
    These constants are fit and determined independently of others in gray body model by fitting data to:
        ln(E_lambda*lambda**5) = a + b/lambda

    ----------------------------------------------------------------------------
    Notes
        Authored by: Michael Braine, Physicist, NIST Gaithersburg
        EMAIL: michael.braine@nist.gov
        October 2022

    ----------------------------------------------------------------------------
    References
        NIST TN 594-13 "The 1973 NBS Scale of Spectral Irradiance"

    ----------------------------------------------------------------------------
    Inputs
        wavelength  - wavelength or array of wavelengths. Expected units: nm
        a           - coefficient in gray body model, related to gray body emissivity with e**a. Needs determined independently by fitting data (see E_planckian function)
        b           - coefficient in gray body model, related to reciprocal of temperature distribution. Needs determined independently by fitting data (see E_planckian function)

    ----------------------------------------------------------------------------
    Returns
        GrayBody - function with independent variable (wavelength) and coefficients (C) as inputs
    """
    return math.e**(a + b/wavelength)/wavelength**5

def UncertaintyFromCovariance(covariance, k=1):
    """
    Calculates uncertainty in coefficients determined from curve-fitting using the fit's covariance matrix.

    ----------------------------------------------------------------------------
    Notes
        Authored by: Michael Braine, Physicist, NIST Gaithersburg
        EMAIL: michael.braine@nist.gov
        October 2022

    ----------------------------------------------------------------------------
    References
        none

    ----------------------------------------------------------------------------
    Inputs
        covariance  - nxn covariance matrix
        k           - level of confidence in returned uncertainty value(s). defaults to k=1

    ----------------------------------------------------------------------------
    Returns
        Uncertainty - n uncertainties
    """
    return np.sqrt(np.diag(covariance))

def GrayBodyCoefficients(wavelength, irradiance):
    """
    Performs curve-fitting on wavelength-irradiance data to generate coefficients and their uncertainties using the gray body model

    ----------------------------------------------------------------------------
    Notes
        Authored by: Michael Braine, Physicist, NIST Gaithersburg
        EMAIL: michael.braine@nist.gov
        October 2022

    ----------------------------------------------------------------------------
    References
        none

    ----------------------------------------------------------------------------
    Inputs
        wavelength  - array of wavelengths. Expected units: nm
        irradiance  - array irradiances. Expected units: W cm^-3 sr^-1
        region      - 1x2 array or tuple of start/stop wavelengths for the interpolation. syntax is [start, stop]. Expected units: nm
        dof         - degrees of freedom for gray body coefficient fitting

    ----------------------------------------------------------------------------
    Returns
        coefficients_GrayBody   - C coefficients in the gray body model
        U                       - uncertainties in the C coefficients
        a                       - coefficient in gray body model, related to gray body emissivity with e**a
        b                       - coefficient in gray body model, related to reciprocal of temperature distribution
    """
    # i_lowerBound, i_upperBound = WavelengthRegionIndex(wavelength, region)

    ab, ab_covariance = curve_fit(E_planckian, wavelength, irradiance, p0=[50, -4800], sigma=irradiance)
    a, b = ab[0], ab[1]
    U_ab = UncertaintyFromCovariance(ab_covariance)

    coefficients_GrayBody, coeff_covariance = curve_fit(__GrayBody__(a, b), wavelength, irradiance, p0=[-1e-4,-1e3,10,-0.01,1], sigma=irradiance)
    U_coeff = UncertaintyFromCovariance(coeff_covariance)
    return coefficients_GrayBody, U_coeff, a, b, U_ab

def GrayBodyInterpolationArb(wavelength, coefficients, a, b):
    """
    Performs interpolation on wavelength-irradiance data to generate coefficients and their uncertainties using the gray body model

    ----------------------------------------------------------------------------
    Notes
        Authored by: Michael Braine, Physicist, NIST Gaithersburg
        EMAIL: michael.braine@nist.gov
        October 2022

    ----------------------------------------------------------------------------
    References
        none

    ----------------------------------------------------------------------------
    Inputs
        region          - 1x2 array or tuple of start/stop wavelengths for the interpolation. syntax is [start, stop]. Expected units: nm
        coefficients    - C coefficients in the gray body model
        a               - coefficient in gray body model, related to gray body emissivity with e**a
        b               - coefficient in gray body model, related to reciprocal of temperature distribution
        step            - step size to perform interpolation

    ----------------------------------------------------------------------------
    Returns
        wavelengths     - array of wavelengths used in interpolation. Units: nm
        irradiances     - array of iraddiances from the interpolation. Units: W cm^-3 sr^-1
    """
    # wavelengths = np.arange(region[0], region[1]+step, step).astype(float)
    return wavelength, __GrayBody__(a, b)(wavelength, *coefficients)


def PeakWavelength(wavelengths, irradiances):
    return wavelengths[np.argmax(irradiances)]

import scipy.constants
def ApparentBBTemp(wavelengths, irradiances):
    """
    Calculates the gray body temperature (apparent black body temperature) using Wein's displacement law

    ----------------------------------------------------------------------------
    Notes
        Authored by: Michael Braine, Physicist, NIST Gaithersburg
        EMAIL: michael.braine@nist.gov
        October 2022

    ----------------------------------------------------------------------------
    References
        none

    ----------------------------------------------------------------------------
    Inputs
        wavelengths  - array of wavelengths. Expected units: nm
        irradiances  - array irradiances. Expected units: W cm^-3 sr^-1

    ----------------------------------------------------------------------------
    Returns
        temperature - apparent black body temperature. Units: K
    """
    return PeakWavelength(wavelengths, irradiances)/1e2/scipy.constants.physical_constants['Wien wavelength displacement law constant'][0]


_LAMBDA0 : float = 450

@dataclass
class SSBUV2(BaseModel):
    ''' 
   SSBUV folliwng NIST model IrradInterPy implementation approach..
    '''
    # NIST-specific attributes
    # _LAMBDA0 : float = 450

    # Interface to NIST IIF Functions.GrayBodyCoefficients() output
    dof : int = 5   # C2, ..., C6 in Huang et al. 1998
    GBcoefficients : np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    GBuncertainty  : np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    GBa : float = 0.0
    GBb : float = 0.0
    abUncertainty   : np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    GBBBtemperature : float = 0.0


    def __post_init__(self) :
        ''' Initialize the subclass by doing the curve_fit plus anything in BaseModel '''
        super().__post_init__() 
    
        # self.wl_fit_limits = np.array([300, 1100]) 
        self.names = ['C2', 'C3', 'C4', 'C5', 'C6', 'a(C0)', 'b(C1)' ]

        # Ensure wl_data and related attributes are initialized before masking
        if not hasattr(self, 'wl_data') or self.wl_data is None:
            raise AttributeError("wl_data must be initialized before NIST model fitting.")
        if not hasattr(self, 'irr_data') or self.irr_data is None:
            raise AttributeError("irr_data must be initialized before NIST model fitting.")
        if not hasattr(self, 'unc_data_abs_k1') or self.unc_data_abs_k1 is None:
            raise AttributeError("unc_data_abs_k1 must be initialized before NIST model fitting.")
        if not hasattr(self, 'unc_data_rel_k2') or self.unc_data_rel_k2 is None:
            raise AttributeError("unc_data_rel_k2 must be initialized before NIST model fitting.")

        # Limit model to masked wavelengths (as per NIST 2011.)
        # i_lowerBound, i_upperBound = IIF.WavelengthRegionIndex(self.wl_data, self.wl_fit_limits)
        # self.wl_data = self.wl_data[i_lowerBound:i_upperBound]
        # self.irr_data = self.irr_data[i_lowerBound:i_upperBound]
        # self.unc_data_abs_k1 = self.unc_data_abs_k1[i_lowerBound:i_upperBound]
        # self.unc_data_rel_k2 = self.unc_data_rel_k2[i_lowerBound:i_upperBound]

        # Fit using NIST IIF
        # (self.GBcoefficients, self.GBuncertainty, self.GBa, self.GBb, self.abUncertainty ) =  \
        #     IIF.GrayBodyCoefficients(self.wl_data, self.irr_data, self.wl_fit_limits, self.dof)
        (self.GBcoefficients, self.GBuncertainty, self.GBa, self.GBb, self.abUncertainty ) =  \
            GrayBodyCoefficients(self.wl_data, self.irr_data)
        self.params = np.concatenate([self.GBcoefficients, [self.GBa], [self.GBb]])
        self.pcov = np.array([])
        self.perr = np.array([])

    
    def model_unc(self, wavelength, nsamples, method = 'covariance', doPlot=False) -> pd.DataFrame:
        match method:
            case 'bootstrap':
                return self.model_unc_bootstrap(wavelength, nsamples, doPlot)
            case 'covariance':
                return self.model_unc_covariance(wavelength, nsamples)
            case _:
                raise ValueError(f"Unknown method: {method}. Use 'bootstrap' or 'covariance.")

    def model_unc_covariance(self, wavelength, _) -> pd.DataFrame:
        '''
        Estimate uncertainty in interpolated points using curve_fit (input parameter) covariance matrix Cp (7 x 7)
        The Jacobian Jp matrix [∂E_i/∂P_j] (E = irradiance, P = fitting params) is calculated. (n_wavelengths x 7)
        Then the confidence intervals are the square root of the diagonal elements of the output covariance matrix Cy:
            Cy = Jp x Cp x Jp'  (n_wavelengths x 1)
        Computationally, for each wavelength index i, compute Cy(i) = Jp(i, :) x Cp x Jp(i,:)' = (1 x 7) X (7 x 7) X (7 x 1)
        See https://stackoverflow.com/questions/77956010/how-to-estimate-error-propagation-on-regressed-function-from-covariance-matrix-u
        and Arras 1998, "An Introduction To Error Propagation: Derivation, Meaning and Examples of Equation Cy = FCF' 
        https://infoscience.epfl.ch/server/api/core/bitstreams/20ca2fc1-b9b7-4316-a30a-938cef8b00a8/content 
        '''
        # def variance(x, Cp, *p):
            
        #     def proxy(q):
        #         irr = self._model_internal(x, *q)
        #         return irr
            
        #     def projection(J):
        #         return J.T @ Cp @ J
        #         # return J @ Cp @ J.T
            
        #     Jp = nd.Gradient(proxy)(*p)
        #     Cy = np.apply_along_axis(projection, 1, Jp)   # Compute output covar major diag
            
        #     return Cy
        
        # # Compute output covariance and k=1 uncertainty in logflux space
        # alpha = 1 - 0.6827 # Corresponds k = 1
        # # alpha = 1 - 0.95 # Corresponds k = 2
        # z = stats.norm.ppf(1 - alpha / 2.)
        # Cy = variance(wavelength, self.pcov, self.params) 
        # sy = np.sqrt(Cy)
        # ci = z * sy
        print(f'{self.__class__.__name__} : Uncertainty estimates using parameter covariances is unimplemented.')
        yhat = self.model(wavelength)
        ci = yhat * 0

        df_interp = pd.DataFrame({
            'wavelength': wavelength,
            'irradiance': yhat,
            'uncertainty': ci,
        })

        return df_interp
        
    def model_unc_bootstrap(self, wavelength, nsamples, doPlot) -> pd.DataFrame:
        '''
        Estimate uncertainty in interpolated points by bootstrapping the mdoel 
        input data. The curve fit is repeated for each bootstrap sample, based 
        on the original lamp and k=2 uncertainty data.  The mean of the interpolated
        irradiance is returned, along with the standard deviation of the
        interpolated irradiance samples.
        '''
        # Create a new random number generator for reproducibility
        rng = np.random.default_rng(42)
        data_bootstrap = []

        # Run bootstrap
        if doPlot:
            fig = go.Figure()
        print(f'Bootstrapping {nsamples} samples...')
        for i in range(nsamples):
            # Add random perturbation irradiance data (more stable results)
            perturbation = rng.normal(0, self.unc_data_abs_k1)
            data_perturbed = self.irr_data + perturbation

            # Fit the model to the perturbed data
            try:
                (GBcoefficients, _, GBa, GBb, _ ) = GrayBodyCoefficients(
                    self.wl_data, 
                    data_perturbed, 
                    )

                GBinterpWavelengths, GBinterpIrradiances = GrayBodyInterpolationArb(
                    wavelength,
                    GBcoefficients,
                    GBa,
                    GBb,
                    )
                data_bootstrap.append(GBinterpIrradiances)

                if doPlot:
                    fig = fig.add_trace(go.Scatter(
                        x=GBinterpWavelengths, y=GBinterpIrradiances, mode='lines',
                        name=f'Bootstrap Sample {i+1}', line=dict(color='grey', width=0.5)
                    ) )
            except RuntimeError:
                continue

        if doPlot:
            fig.show('browser')  # Show the bootstrap samples
        irr_samples = np.array(data_bootstrap)  # shape: (n_samples, len(wwavlength))

        # Compute statistics
        irr_mean = np.nanmean(irr_samples, axis=0)
        irr_std = np.nanstd(irr_samples, axis=0)

        # Package interpolated results
        df_interp = pd.DataFrame({
            'wavelength': GBinterpWavelengths,
            'irradiance': irr_mean,
            'uncertainty': irr_std
        })

        return df_interp


    def model(self, wavelength):
        '''Compute model results transformed to irradiance'''

        # Don't use self.wl_fit_limits, allow users to evaluate GrayBody outside the "fitted range, if they want
        # NOTE: assume interpolation wavelengths are passed in as evenly spaced.
        limits = [wavelength[0], wavelength[-1]]
        step = wavelength[1] - wavelength[0]

        check_for_zero = np.arange(limits[0], limits[1]+step, step) - wavelength
        zero_count = np.sum(np.isclose(check_for_zero, 0., rtol=1e-9, atol=1e-9))
        if zero_count != wavelength.shape[0] :
            print(f'Computed wavelength array in {self.__class__.__name__} does not match input wavelength array')
        
        # Evaluate NIST model at wavelengths =  = np.arange(limits[0], limits[1]+step, step)
        _, GBinterpIrradiances = GrayBodyInterpolationArb(
            wavelength,  
            self.GBcoefficients,
            self.GBa,
            self.GBb,
        )

        # Use limited wavelengths to compute GBBB temperature
        # mask = (wavelength >= self.wl_fit_limits[0]) & (wavelength <= self.wl_fit_limits[1]) # Works ok.
        # self.GBBBtemperature = IIF.ApparentBBTemp(wavelength[mask], GBinterpIrradiances[mask])
        self.GBBBtemperature = ApparentBBTemp(wavelength, GBinterpIrradiances)

        return GBinterpIrradiances
    
    def residuals(self) :
        # i_lowerBound, i_upperBound = IIF.WavelengthRegionIndex(
        #     self.wl_data, self.wl_fit_limits
        # )
        # return 100 * (IIF.GrayBody(self.wl_data[i_lowerBound:i_upperBound], self.GBa, self.GBb, self.GBcoefficients) - self.irr_data[i_lowerBound:i_upperBound])/self.irr_data[i_lowerBound:i_upperBound]
        return 100 * (GrayBody(self.wl_data, self.GBa, self.GBb, self.GBcoefficients) - self.irr_data)/self.irr_data
        
