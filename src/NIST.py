from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from BaseModel import BaseModel

@dataclass
class NIST(BaseModel):
    ''' 
    NIST model is constrainted to being fit in specific regions
    See Yoon, H.W and Gibson, C.E.
    "Spectral Irradiance Calibrations", NIST Special Publication 250-89,2011
    '''
    # NIST-specific attributes
    wl_fit_limits : np.ndarray = field(default_factory=lambda: np.array([350, 800])) # NIST model fit limits in nm
    _LAMBDA0 : np.ndarray = field(default_factory=lambda: np.array([350, 800])) 


    def __post_init__(self) :
        ''' Initialize the subclass by doing the curve_fit plus anything in BaseModel '''
        super().__post_init__() 
    
        # self.wl_fit_limits = np.array([300, 1100]) 
        self._LAMBDA0 = self.wl_fit_limits

        self.names = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'T' ]

        # Initial guesses for parameters: A0,...,A5, T
        self.p0 = np.array([10,-1000,-1e-4,-1e3,10,-0.01,1])  # NIST

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
        data_bootstrap = []

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

                # pred = self.untransform(wavelength, log_flux)  # shape: (len(wavelength),)
                # fig = fig.add_trace(go.Scatter(
                #     x=wavelength, y=pred, mode='lines',
                #     name=f'Bootstrap Sample {_+1}', line=dict(color='grey', width=0.5)
                # ) )
            except RuntimeError:
                continue

        # fig.show()  # Show the bootstrap samples
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
