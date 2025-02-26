"""
Example of creating systematic band for histogram
"""
import collinearw
import numpy as np

# create a histogram object
hist = collinearw.core.Histogram("nJet", 5, 0, 5)

# example of creating systematic band container
exp_band = collinearw.core.SystematicBand(
    "Jet Energy",  # name of the band container
    "experimental",  # type of the band container
    hist.shape,  # shape should match the histogram
)

# create some random multiplicative/fractional uncertainties
temp_up, temp_down = np.zeros(hist.shape), np.zeros(hist.shape)
exp_band.add_component("up", "Jet_NP_1", temp_up + 0.1)
exp_band.add_component("down", "Jet_NP_1", temp_up + 0.1)
exp_band.add_component("up", "Jet_NP_2", temp_up + 0.2)
exp_band.add_component("down", "Jet_NP_2", temp_up + 0.2)

# add the band into the histogram
hist.update_systematic_band(exp_band)

print(f"{hist.systematic_band=}, {type(hist.systematic_band)}")
print(f"{hist.systematic_band['Jet Energy'].components=}")
