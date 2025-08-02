from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy import stats
import numdifftools as nd
import plotly.graph_objects as go
from BaseModel import BaseModel
import sys
sys.path.insert(1, '../IrradInterPy/src')
import Functions.IrradianceInterpolationFuncs as IIF

@dataclass
class NISTIIF(BaseModel):
    ''' 
    NIST model using usnistgov IrradInterPy implemention.

    See https://github.com/usnistgov/IrradInterPy
    You must git clone https://github.com/usnistgov/IrradInterPy.git into a peer directory

    Model self-limits to wl_fit_limits for fitting, although interpolation is allowed outside this range.
    Due to IrradInterPy internals (see FFI.GrayBodyInterpolation), interpolated wavelength basis must be uniform (lower, upper, step)
    (This must be rectified to be useful for interpolation on to general radiometer wavelength basis, e.g., Zeiss MMS/CGS)

    See Yoon, H.W and Gibson, C.E. "Spectral Irradiance Calibrations", NIST Special Publication 250-89,2011
    '''
    # NIST-specific attributes
    wl_fit_limits : np.ndarray = field(default_factory=lambda: np.array([350, 800])) # NIST model fit limits in nm

    # Interface to NIST IIF Functions.GrayBodyCoefficients() output
    dof : int = 5
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
        self.names = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'a', 'b' ]

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
        i_lowerBound, i_upperBound = IIF.WavelengthRegionIndex(self.wl_data, self.wl_fit_limits)
        self.wl_data = self.wl_data[i_lowerBound:i_upperBound]
        self.irr_data = self.irr_data[i_lowerBound:i_upperBound]
        self.unc_data_abs_k1 = self.unc_data_abs_k1[i_lowerBound:i_upperBound]
        self.unc_data_rel_k2 = self.unc_data_rel_k2[i_lowerBound:i_upperBound]

        # Fit using NIST IIF
        (self.GBcoefficients, self.GBuncertainty, self.GBa, self.GBb, self.abUncertainty ) =  \
            IIF.GrayBodyCoefficients(self.wl_data, self.irr_data, self.wl_fit_limits, self.dof)
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

        limits = [wavelength[0], wavelength[-1]]
        step = wavelength[1] - wavelength[0]

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
                (GBcoefficients, _, GBa, GBb, _ ) = IIF.GrayBodyCoefficients(
                    self.wl_data, 
                    data_perturbed, 
                    self.wl_fit_limits, 
                    self.dof
                    )

                GBinterpWavelengths, GBinterpIrradiances = IIF.GrayBodyInterpolation(
                    limits,
                    GBcoefficients,
                    GBa,
                    GBb,
                    step,
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
        _, GBinterpIrradiances = IIF.GrayBodyInterpolation(
            limits,  
            self.GBcoefficients,
            self.GBa,
            self.GBb,
            step,
        )

        # Use limited wavelengths to compute GBBB temperature
        mask = (wavelength >= self.wl_fit_limits[0]) & (wavelength <= self.wl_fit_limits[1]) # Works ok.
        self.GBBBtemperature = IIF.ApparentBBTemp(wavelength[mask], GBinterpIrradiances[mask])

        return GBinterpIrradiances
    
    def residuals(self) :
        i_lowerBound, i_upperBound = IIF.WavelengthRegionIndex(
            self.wl_data, self.wl_fit_limits
        )
        return 100 * (IIF.GrayBody(self.wl_data[i_lowerBound:i_upperBound], self.GBa, self.GBb, self.GBcoefficients) - self.irr_data[i_lowerBound:i_upperBound])/self.irr_data[i_lowerBound:i_upperBound]
        
