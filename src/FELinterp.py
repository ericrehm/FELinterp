import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
from punpy.mc.mc_propagation import MCPropagation
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class BaseModel:

    ''' 
        Abstract base clase for interpolation (and related uncertainty) models 
        Constructor must supply the basic lamp cal data: wavelength, irradiance, k=2 rel uncertainty (%)
    '''
    wl_data : np.ndarray 
    irr_data: np.ndarray
    unc_data_rel_k2: np.ndarray   # Optronic Lamp uncertainty data is in this form (k=2 %uncert)

    # Initialized class variables that are not required in the constructor
    unc_data_abs_k1: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    k: float = 2
    p0     : np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    params : np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    pcov   : np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    perr   : np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    names  : List[str]  = field(default_factory=list)
    
    # Methods
    def __post_init__(self) :
        self.unc_data_abs_k1 = ((self.unc_data_rel_k2/self.k)/100) * self.irr_data

    def __str__(self):
        return f"{self.__class__.__name__}"

    def transform(self, wl: np.ndarray, irr: np.ndarray) -> np.ndarray:
        irr_transformed = irr
        return irr_transformed
    
    def untransform(self, wl: np.ndarray, irr_transformed: np.ndarray) -> np.ndarray:
        irr = irr_transformed
        return irr
    
    def model(self, wl: np.ndarray):
        raise NotImplementedError("Subclasses must implement model() method")

    def print_model(self):
        # Print fitted parameters
        print(f"{str(self)}")
        for name, value in zip(self.names, self.params):
            print(f"{name} = {value:.3e}")
        
    def residuals(self) -> np.ndarray:
        '''Run model at input wavelengths to calculate residuals'''
        I_predicted = self.model(self.wl_data)
        RPD = 100*(I_predicted.astype(float) - self.irr_data.astype(float))/self.irr_data.astype(float) 
        return RPD

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({'wavelength': self.wl_data, 
                             'irradiance' : self.irr_data,
                             'uncertainty_rel' : self.unc_data_rel_k2, 
                             'uncertainty_abs_k1' : self.unc_data_abs_k1
                             })

    def plotly_unc(self, df_interp : pd.DataFrame, k : int = 2, fig = None) -> go.Figure:
        df = self.to_dataframe()

        low_band = df_interp['irradiance'] - k * df_interp['uncertainty']
        high_band = df_interp['irradiance'] + k * df_interp['uncertainty']

        # # Plotly the plot
        if (fig is None):
            fig = go.Figure()

        # Original data
        fig.add_trace(go.Scatter(
            x=df['wavelength'], y=df['irradiance'],
            mode='markers', name='Original Data',
            error_y=dict(type='data', array=df['uncertainty_abs_k1']*2, visible=True),
            marker=dict(color='blue', size=6)
        ))

        # Interpolated fit
        fig.add_trace(go.Scatter(
            x=df_interp['wavelength'], y=df_interp['irradiance'],
            mode='lines', name='Propagated mean uncert.', line={'color':'grey','dash':'dot'}
        ))

        # Uncertainty band (±kσ)
        fig.add_trace(go.Scatter(
            x=np.concatenate([df_interp['wavelength'], df_interp['wavelength'][::-1]]),
            y=np.concatenate([low_band, high_band[::-1]]),
            fill='toself', fillcolor='rgba(0,0,255,0.2)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=True, name=f'±{k}σ Band'
        ))

        fig.update_layout(
            title=f'Spectral Irradiance Fit with k={k} Uncertainty',
            xaxis_title='Wavelength (nm)',
            yaxis_title='Irradiance (W/cm²/nm)',
            template='plotly_white'
        )

        return fig
    

    def plotly_rel_residuals(self, df_interp : pd.DataFrame, fig = None) -> go.Figure:
        RPD = self.residuals()
        
        if (fig is None):
            fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.wl_data, y=RPD, mode='markers+lines',
            name='Residuals (RPD)', marker={'color':'blue'},
            line=dict(dash='dot')
        ))
        fig.update_layout(
            # title=f'$\\text{{FEL Lamp Irradiance RPD using theModel Method }}(\lambda_0 = {self._LAMBDA0})$',
            title=f'$\\text{{FEL Lamp Irradiance RPD}}$',
            xaxis_title='Wavelength (nm)',
            yaxis_title='100 x Relative Deviation (%)',
            legend=dict(x=0.01, y=0.99),
            template='plotly',
            yaxis=dict(range=[-0.6, 0.6])  # Set y-axis limits
        )
        return fig

@dataclass
class NIST(BaseModel):
    ''' 
    NIST model is constrainted to being fit in specific regions
    See Yoon, H.W and Gibson, C.E.
    "Spectral Irradiance Calibrations", NIST Special Publication 250-89,2011
    '''
    # NIST-specific attributes
    wl_fit_limits : np.ndarray = field(default_factory=lambda: np.array([300, 1100])) # NIST model fit limits in nm
    # wl_fit_limits : np.ndarray = np.array([300, 1100])  # Default fit limits
    _LAMBDA0 : np.ndarray = field(default_factory=lambda: np.array([300, 1100])) 


    def __post_init__(self) :
        ''' Initialize the subclass by transforming data, doing the curve_fit plus anything in BaseModel '''
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
        return self.__model_internal(wavelength, *self.params)


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

@dataclass
class WhiteSpline(BaseModel):

    spline: InterpolatedUnivariateSpline = field(init=False)
    _LAMBDA0 : float = 0  

    def __post_init__(self) :
        ''' Initialize the subclass by transforming data, doing the curve_fit plus anything in BaseModel '''
        super().__post_init__() 
    
        self.spline = InterpolatedUnivariateSpline(self.wl_data, self.irr_data, ext=2, check_finite=True)

        # self.perr = np.sqrt(np.diag(self.pcov))   # stdev of model params

    def print_model(self):
        print(f"{str(self)}")
        print(self.spline)
        return

    def model(self, wavelength):
        '''Compute model results: irradiance at wavelength'''
        xi = wavelength.ravel()
        return self.spline(xi)
    
    def model_unc_white(self, wavelength) -> pd.DataFrame:
        '''
            Estimate uncertainty in interpolated points 
            '''
        # Interpolate (x,y) to new xi basis yielding yi
        # MATLAB: pp = spline(x,y)
        #         yi = ppval(pp, xi)
        # x = x.to_numpy()
        # y = y.to_numpy()
        # xi = xi.to_numpy()
        N = len(self.irr_data)
        Ni = len(wavelength)

        # Original data
        x = self.wl_data
        y = self.irr_data
        uy = self.unc_data_abs_k1

        # Interpolated data
        xi = wavelength.ravel()
        yi = self.model(wavelength)

        # Calculate spline weights for each knot (1, 0, ...), (0, 1, ...), etc.
        yeye = np.eye(N)
        ppFx = []
        for i in range(0,N) :
            ppFx.append(InterpolatedUnivariateSpline(x, yeye[i,:]))

        # Calculate uncertainties uyi at interpolated points
        #
        # Essentially, a spline-weighted sum of the known uncertainties uy
        # where the splines are evaluated at each xi
        uyi = np.zeros(Ni)
        for j in range(0, Ni) :
            for spline,i in zip(ppFx, range(0,N)) :
                uyi[j] = uyi[j] + (uy[i]**2)*(spline(xi[j])**2)  # Be careful w indexing!

        uyi = np.sqrt(uyi)

        # Package interpolated results
        df_interp = pd.DataFrame({
            'wavelength': wavelength,
            'irradiance': yi,
            'uncertainty': uyi
        })

        return df_interp
    
    
# Local functions

def readOptronicLampData(lampSN) :
    lampBaseDir = Path('/Users/ericrehm/src/FELinterp/lamps/Optronic')
    lampSNDir   = re.sub(r'F(\d+)', r'F-\1', lampSN)
    std_files = list(Path(lampBaseDir, lampSNDir).glob(f'{lampSN}_??.std'))
    if not std_files:
        raise FileNotFoundError(f"No file matching pattern '{lampSN}_??.std' found in {Path(lampBaseDir, lampSNDir)}")
    lampPath = std_files[0]
    lampUncPath = Path(lampBaseDir, lampSNDir,f'{lampSN}_k2uncertainty.dat')

    # --- Read the Optronic Labs FEL lamp .std file ---
    try:
        print(f'Reading {lampPath}')
        df1 = pd.read_csv(lampPath, header=None, names=['wavelength', 'irradiance'], skiprows = 1)
        df1 = df1[df1.columns[:2]]  # Just in case extra columns got read
    except FileNotFoundError:
        print(f"Error: The file '{lampPath}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred while reading '{lampPath}': {e}")
        
    # --- Read the Optronic Labs FEL lamp k=2 uncertainty file ---
    try:
        print(f'Reading {lampUncPath}')
        df2 = pd.read_csv(lampUncPath, header=None, sep='\t', names=['wavelength', 'uncertainty_rel'], skiprows = 1)
    except FileNotFoundError:
        print(f"Error: The file '{lampUncPath}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred while reading '{lampUncPath}': {e}")
   
    df = pd.merge(df1, df2, on='wavelength')
    return df

def main():

    lampSN = 'F1711'
    # lampSN = 'F1738'
    # lampSN = 'F1739'
    nsamples = 50

    df = readOptronicLampData(lampSN)

    # # Read 1 nm lamp FIT file interpolated by SISS
    # lampPath = lampBaseDir / Path('F-1711/F1711.FIT')
    # dff = pd.read_csv(lampPath, header=None, sep='\\s+', names=['wavelength', 'irradiance'], skiprows = 3)

    # Create theModel model and print fitted parameters
    # theModel = SSBUV(df.wavelength.values, df.irradiance.values, df.uncertainty_rel.values)
    # theModel = SSBUVw0(df.wavelength.values, df.irradiance.values, df.uncertainty_rel.values)
    # theModel = NIST(df.wavelength.values, df.irradiance.values, df.uncertainty_rel.values, wl_fit_limits=np.array([350, 800]) )
    theModel = WhiteSpline(df.wavelength.values, df.irradiance.values, df.uncertainty_rel.values)
    theModel.print_model()

    # Interpolate at user-defined wavelengths 
    # (Shows how to use the model without knowledge of any internals)
    wavelengths = df.wavelength.values
    w_interp = np.linspace(wavelengths.min(), wavelengths.max(), 1000)  # Arbitrary wl grid
    # w_interp = np.linspace(250.0, 1100.0, (1100-250)+1)  # Arbitrary wl grid
    I_interp = theModel.model(w_interp)

    # Model uncertainties via MCPropagation at interpolated wavelengths (not perfect yet)
    # uncEstStr = 'MCPropagation interp. uncert.'  # Uncomment to use MCPropagation
    # interp_df = theModel.model_unc_mc(w_interp, nsamples=10000)
    # uncEstStr = 'bootstrap'    # Use bootstrap to estimate uncertainties
    # interp_df = theModel.model_unc_bootstrap(w_interp, nsamples=nsamples)
    uncEstStr = 'WhiteSpline'    # Use bootstrap to estimate uncertainties
    interp_df = theModel.model_unc_white(w_interp)  # Use WhiteSpline model to estimate uncertainties
    # Plot original data and modeled mean spectrum + uncertainties evaluated at interpolated 
    # wavelengths then plot residuals
    fig1 = theModel.plotly_unc(interp_df)
    fig2 = theModel.plotly_rel_residuals(interp_df)

    # Add the diretly interpolated data
    fig1.add_trace(go.Scatter(
        x=w_interp, y=I_interp, mode='lines',
        name='Interpolated Fit', line=dict(color='red')
    ))

    # # Add SISS 1 nm .FIT file
    # fig1.add_trace(go.Scatter(
    #     x=dff.wavelength, y=dff.irradiance/1e6, mode='lines',
    #     name='SISS Fit', line=dict(color='black', dash='dash')
    # )).show()

    # Put the two figures on subplots 
    fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], vertical_spacing=0.02, shared_xaxes=True, subplot_titles=
                        (f'$\\text{{FEL Lamp {lampSN} Irradiance fit with {str(theModel)} Method }}(\lambda_0 = \\text{{{str(theModel._LAMBDA0)} }}nm)\\text{{, Uncertainty: {uncEstStr} method, N={nsamples}}}$', ''))
    
    # Populate the subplots from the two figures (plotly feature/hack)
    for itrace in fig1.data: 
        fig.add_trace(itrace, row=1, col=1)
    for itrace in fig2.data: 
        fig.add_trace(itrace, row=2, col=1)
    
    # Update the axis labels and ranges
    fig.update_yaxes(title_text='$\\text{Radiance  }({\\mu}W cm^{-2} nm^{-1})$', range=[-1e-6, 25e-6], row=1, col=1)
    fig.update_xaxes(title_text='$\\text{Wavelength  }(nm)$', row=2, col=1)
    fig.update_xaxes(range=[w_interp.min()-10, w_interp.max()+10])
    fig.update_yaxes(title_text='$\\text{100 x Relative Deviation}$', range=[-0.6 ,0.6], row=2, col=1)
    fig.show('browser')  # Show the plot in a browser

if __name__ == '__main__' :
    main()
