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

