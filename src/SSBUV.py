from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from punpy.mc.mc_propagation import MCPropagation
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

    def model_unc_mc(self, wavelength, nsamples=10000) -> pd.DataFrame:
        '''
        Estimate uncertainty in interpolated points by computing transformed 
        data and transformed uncertainties, resampling transforded data, and 
        then error propagation.
        
        This is different than resampling model parameters based pcov (the 
        covariance matrix of the fit) -- difficult because pcov is 
        ill-conditioned. This means SSBUV is "twitchy", and some error 
        propagation approaches don't work well as a result
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
        irradiance_pred = np.mean(irr_samples, axis=1)
        irradiance_std  = np.std(irr_samples, axis=1)
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
    
    def model_unc_bootstrap(self, wavelength, nsamples=1000) -> pd.DataFrame:
        '''
        Estimate uncertainty in interpolated points by bootstrapping the mdoel 
        input data. The curve fit is repeated for each bootstrap sample, based 
        on the original lamp and k=2 uncertainty data.  The mean of the interpolated
        irradiance is returned, along with the standard deviation of the
        interpolated irradiance samples.
        This may be a more robust approach than MCPropagation method above.
        '''
        # Create a new random number generator for reproducibility
        rng = np.random.default_rng(42)
        log_flux_bootstrap = []

        # fig = go.Figure()

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

                # pred = self.untransform(wavelength, log_flux)  # shape: (len(wavelength),)
                # fig = fig.add_trace(go.Scatter(
                #     x=wavelength, y=pred, mode='lines',
                #     name=f'Bootstrap Sample {_+1}', line=dict(color='grey', width=0.5)
                # ) )
            except RuntimeError:
                continue

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

