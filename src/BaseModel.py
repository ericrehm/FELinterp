from dataclasses import dataclass, field
from typing import List
import pandas as pd
import numpy as np
import plotly.graph_objects as go


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

