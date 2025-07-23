from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
from BaseModel import BaseModel


@dataclass
class WhiteSpline(BaseModel):
    ''' 
    WhiteSpline model, White et al. 2017, "Propagation of Uncertainty and Comparison of Interpolation Schemes"
    '''

    spline: make_interp_spline = field(init=False)
    _LAMBDA0 : float = 0  

    def __post_init__(self) :
        ''' Initialize the subclass by doing the curve_fit plus anything in BaseModel '''
        super().__post_init__() 
    
        self.spline = make_interp_spline(self.wl_data, self.irr_data, k=3, check_finite=True)

        # self.perr = self.spline.get_residual() # stdev of model params

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
            ppFx.append(make_interp_spline(x, yeye[i,:], k=3))

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
    
    
#