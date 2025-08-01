from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import numdifftools as nd
import plotly.graph_objects as go
from BaseModel import BaseModel

@dataclass
class NIST(BaseModel):
    ''' 
    NIST model is constrainted to being fit in specific regions
    See Yoon, H.W and Gibson, C.E.
    "Spectral Irradiance Calibrations", NIST Special Publication 250-89,2011
    '''
    # NIST-specific attributes
    wl_fit_limits : np.ndarray = field(default_factory=lambda: np.array([350, 1100])) # NIST model fit limits in nm
    _LAMBDA0 : np.ndarray = field(default_factory=lambda: np.array([350, 800])) 


    def __post_init__(self) :
        ''' Initialize the subclass by doing the curve_fit plus anything in BaseModel '''
        super().__post_init__() 
    
        # self.wl_fit_limits = np.array([300, 1100]) 
        self._LAMBDA0 = self.wl_fit_limits

        self.names = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'T' ]

        # Initial guesses for parameters: A0,...,A5, T
        self.p0 = np.array([10,-1000,-1e-4,-1e3,10,-0.01,1])  # NIST

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
        mask = (self.wl_data >= self.wl_fit_limits[0]) & (self.wl_data <= self.wl_fit_limits[1]) # Works ok.
        self.wl_data = self.wl_data[mask]
        self.irr_data = self.irr_data[mask]
        self.unc_data_abs_k1 = self.unc_data_abs_k1[mask]
        self.unc_data_rel_k2 = self.unc_data_rel_k2[mask]

        # Fit using weighted, non-linear least squares
        (self.params, self.pcov, _, _, _) = curve_fit(        # type: ignore
            self._model_internal,
            self.wl_data,
            self.irr_data,
            sigma=self.unc_data_abs_k1,
            absolute_sigma=True,
            p0 = self.p0,
            maxfev = 10000,
            method = 'lm',
            full_output=True
        )
        self.perr = np.sqrt(np.diag(self.pcov))   # stdev of model params

    def _model_internal(self, wavelength, *params):

        '''Compute model results in log flux, passing params (used by curve_fit during fitting)'''

        A0, A1, A2, A3, A4, A5, T = params
        C2 = 0.01438777 # Second radiation constant C2 in m⋅K 

        # Pre-allocate model result F
        F = np.zeros_like(wavelength, dtype=float)

        # wavelength = wavelength / 2e4  #Scaling necessary to achieve good fit. Overflow?
        # F = (A0 + A1*wavelength + A2*wavelength**2+ A3*wavelength**3 + A4*wavelength**4)/(wavelength**5)*np.exp(A5 + C2/(wavelength*T))
        
        wavelength = wavelength / 1e4  # Scaling necessary to achieve good fit. Overflow?  Better fit than
        F = (A0 + A1*wavelength + A2*wavelength**2+ A3*wavelength**3 + A4*wavelength**4)/(wavelength**5)*np.exp(A5 + T/(wavelength))

        return F
    
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
        NOTE:  DOES NOT PRODUCE REASONABLE UNCERTAINTY ESTIMATES.  USE 'bootstrap' method instead.
        
        Estimate uncertainty in interpolated points using curve_fit (input parameter) covariance matrix Cp (7 x 7)
        The Jacobian Jp matrix [∂E_i/∂P_j] (E = irradiance, P = fitting params) is calculated. (n_wavelengths x 7)
        Then the confidence intervals are the square root of the diagonal elements of the output covariance matrix Cy:
            Cy = Jp x Cp x Jp'  (n_wavelengths x 1)
        Computationally, for each wavelength index i, compute Cy(i) = Jp(i, :) x Cp x Jp(i,:)' = (1 x 7) X (7 x 7) X (7 x 1)
        See https://stackoverflow.com/questions/77956010/how-to-estimate-error-propagation-on-regressed-function-from-covariance-matrix-u
        and Arras 1998, "An Introduction To Error Propagation: Derivation, Meaning and Examples of Equation Cy = FCF' 
        https://infoscience.epfl.ch/server/api/core/bitstreams/20ca2fc1-b9b7-4316-a30a-938cef8b00a8/content 
        '''
        def variance(x, Cp, *p):
            
            def proxy(q):
                irr = self._model_internal(x, *q)
                return irr
            
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
        ci = z * sy
        yhat = self.model(wavelength)

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

        if doPlot:
            fig = go.Figure()

        # Run bootstrap
        print(f'Bootstrapping {nsamples} samples...')
        for _ in range(nsamples):
            # Add random perturbation to the log flux data (more stable results)
            perturbation = rng.normal(0, self.unc_data_abs_k1)
            data_perturbed = self.irr_data + perturbation

            # Alternatively, perturb the irradiance data
            # perturbation = rng.normal(0, self.unc_data_abs_k1)
            # irr_data_perturbed = self.irr_data + perturbation
            # log_flux_data_perturbed = self.transform(self.wl_data, irr_data_perturbed)

            # Fit the model to the perturbed data
            try:
                (popt_i, _, _, _, _) = curve_fit(                    # type: ignore
                            self._model_internal,
                            self.wl_data,
                            data_perturbed,
                            p0 = self.p0,
                            maxfev = 10000,
                            method = 'lm',
                            full_output=True
                )

                irr = self._model_internal(wavelength, *popt_i)
                data_bootstrap.append(irr)

                if doPlot:
                    pred = irr  # shape: (len(wavelength),)
                    fig = fig.add_trace(go.Scatter(
                        x=wavelength, y=pred, mode='lines',
                        name=f'Bootstrap Sample {_+1}', line=dict(color='grey', width=0.5)
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
            'wavelength': wavelength,
            'irradiance': irr_mean,
            'uncertainty': irr_std
        })

        return df_interp


    def model(self, wavelength):
        '''Compute model results transformed to irradiance'''
        return self._model_internal(wavelength, *self.params)
