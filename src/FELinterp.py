import re
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from WhiteSpline import WhiteSpline
from NIST import NIST
from SSBUV import SSBUV, SSBUVw0 
 


# Local functions

def readOptronicLampData(lampSN) :
    '''
    Read Optronic Labs FEL lamp data from .std and k=2 uncertainty files.
    
    OLFEL-M 1000-Watt NIST-Traceable Spectral Irradiance Standard (250 - 1100 nm)
    The following reported calibration values are covered under our ISO 17025 scope:
    250 - 400 nm in (10) nm increments, 450 nm, 500 nm, 555 nm,
    600 nm, 654.6 nm, 700 nm, 800 nm, 900 nm, 1050 nm, 1100 nm
    https://www.solarlight.com/product/ol-fel-m-irradiance-standard-250-1100-nm?tab=description
    Spectral Irradiance (Nominal): @ 250 nm: 0.03 μW/cm²nm, @ 1000 nm: 25μW/cm²nm

    '''

    lampBaseDir = Path('/Users/ericrehm/src/FELinterp/lamps/Optronic')
    lampSNDir   = re.sub(r'F(\d+)', r'F-\1', lampSN)
    std_files = list(Path(lampBaseDir, lampSNDir).glob(f'{lampSN}_??.std'))
    if not std_files:
        raise FileNotFoundError(f"No file matching pattern '{lampSN}_??.std' found in {Path(lampBaseDir, lampSNDir)}")
    lampPath = std_files[0]
    lampUncPath = Path(lampBaseDir, lampSNDir,f'{lampSN}_k2uncertainty.dat')

    # --- Read the Optronic Labs FEL lamp .std file with units: W/(cm^2 nm) ---
    try:
        print(f'Reading {lampPath}')
        df1 = pd.read_csv(lampPath, header=None, names=['wavelength', 'irradiance'], skiprows = 1)
        df1 = df1[df1.columns[:2]]  # Just in case extra columns got read
    except FileNotFoundError:
        print(f"Error: The file '{lampPath}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred while reading '{lampPath}': {e}")
        
    # --- Read the Optronic Labs FEL lamp k=2 relative uncertainty file (with units percent %) ---
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
    # theModel = SSBUV(df.wavelength.to_numpy(), df.irradiance.to_numpy(), df.uncertainty_rel.to_numpy())
    # theModel = SSBUVw0(df.wavelength.values, df.irradiance.values, df.uncertainty_rel.values)
    # theModel = NIST(df.wavelength.to_numpy(), df.irradiance.to_numpy(), df.uncertainty_rel.to_numpy(), wl_fit_limits=np.array([350, 800]) )
    theModel = WhiteSpline(df.wavelength.to_numpy(), df.irradiance.to_numpy(), df.uncertainty_rel.to_numpy())
    theModel.print_model()

    # Interpolate at user-defined wavelengths 
    # (Shows how to use the model without knowledge of any internals)
    wavelengths = df.wavelength.values
    w_interp = np.linspace(wavelengths.min(), wavelengths.max(), 1000)  # Arbitrary wl grid
    # w_interp = np.linspace(250.0, 1100.0, (1100-250)+1)  # Arbitrary wl grid
    I_interp = theModel.model(w_interp)

    # Model uncertainties via MCPropagation at interpolated wavelengths (not perfect yet)
    # uncEstStr = 'MCPropagation'  # Uncomment to use MCPropagation    
    uncEstStr = 'bootstrap'    # Use bootstrap to estimate uncertainties
    # uncEstStr = 'WhiteSpline'    # Use bootstrap to estimate uncertainties
    interp_df = theModel.model_unc(w_interp, nsamples=nsamples, method=uncEstStr)  
    
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
