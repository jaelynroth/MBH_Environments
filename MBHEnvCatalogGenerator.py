"""LISA AstroWG - MBH Environments - Catalog generation script - v1.0

This script used the script written by Matteo Bonetti & Luke Zoltan Kelley as a blueprint. 

Catalog v1.0 is primarily meant to collect data from datasets on
(i) the environments surrounding MBH (numerical) mergers


Authors
-------
- Matteo Bonetti : matteo.bonetti@unimib.it
- Luke Zoltan Kelley : lzkelley@berkeley.edu
- John Regan: john.regan@mu.ie

- Structure
-------
This script has four main parts:
    - part A [MODIFY]       : functions that need input from users to collect information about the specific models.
                            Please carefully read the docstring of the function 'input_data()' that you find below.
    - part B [DO NOT MODIFY]: data structure for the hdf5 file
    - part C [DO NOT MODIFY]: functions to produce and validate hdf5 files. 
    - part D [DO NOT MODIFY]: main routine.

----
- Validation
    - Make sure numbers of elements all match
    - Even when not doing units checks, make sure values are sane (e.g. all positive, nonzero; etc) - DONE

---
- Simple usage:

python3 MBHEnvCatalogGenerator_<YourSimulationDataset>.py <your_command_line_arguments>



"""
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

# relevant packages, DO NOT REMOVE THEM
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys
import pickle
import glob

__VERSION__ = '1.0' # catalog version
DEBUG = False
DEF_FILENAME = "example-sim"
np.random.seed(5)

############################################################################################
############################################################################################
############################################################################################
#### PART A: CHANGES NEEDED FROM USERS #####################################################
############################################################################################
############################################################################################
############################################################################################

#You'll need to change this to your simulation
SIMULATION="SEEDZ"

def input_data():
    '''
    This function collects the necessary information to produce the MBH Env Catalog

    Users should perform the following actions:
    1) edit the 'metadata' dictionary in this function to provide the specific information concerning a certain model;
    2) edit the 'get_binary_information' function in order to populate the numpy arrays with information from your datasets

    IMPORTANT: If a field does not apply to your specific model (e.g. spatial resolution for EPS SAMs), set it to 'np.nan'.

    NOTE: we require the number density of events. This is a meaninful quantity for EPS SAMs, for cosmological simulations
    and SAMs based on dark matter merger trees this is just 1/V_box 

    All parts that need to be modified by the user are enclosed into starred blocks like this:
    #************************************************#
    #************************************************#
    ...
    ...
    ...
    ...
    #************************************************#
    #************************************************#
        
    All the other functions should be left unchanged.

    Returns: 2 dictionaries -> metadata, mbhenv
    '''

    ###########################################################
    ####### CHANGE THE CODE BELOW #############################
    ###########################################################
    #*********************************************************#
    #*********************************************************#

    metadata = {
        # ---- Header data - identifying information for the dataset
        # REQUIRED
        'SimulationName': SIMULATION,
        'SimulationVersion': 'LowRes',
        'ModelType': 'Hydro',
        'Version': __VERSION__,
        'Date': str(datetime.datetime.now()),
        'Contributor': ["John Regan", "Daxal Mehta", "Lewis Prole"],
        'Email': ["john.regan@mu.ie"],
        'Principal': ["John Regan"],
        'Reference': ["DOI1", "DOI2"],
        # OPTIONAL:
        'Website': ["https://www.github.com/mbonetti90/MBHCatalogs"],

        # ---- Model parameters - metadata specification for simulation(s) used to construct catalog
        # REQUIRED
        'HubbleConstant': 70, #km s^-1 Mpc^-1
        'OmegaMatter': 0.3,
        'OmegaLambda': 0.7,
        'BoxSize': 1e0, # cMpc h^-1 (comoving)
        'MinBHSeedMass': 5e3, #M_sun
        'MinRedshift': 10,
        'MaxRedshift': 20,
        'StellarMassResolution': 1e3, # M_sun
        'DarkMatterMassResolution': 1e5, # M_sun
        'SpatialResolution': 5, # pc h^-1 (comoving)
        # OPTIONAL:
        'MinimumDarkMatterHaloMass': 1e6, # M_sun
        'GasMassResolution': 1e2, # M_sun
    }
    #*********************************************************#
    #*********************************************************#

    ###########################################################
    #### DO NOT CHANGE FUNCTION CALLS AND RETURN ##############
    ###########################################################


    mbhenv = get_binary_information(metadata)


    return metadata, mbhenv

############################################################################################

def get_binary_information(metadata):

    '''
    Function to collect properties of binaries assuming no-delay models.
    In principle we can include delays as the MBHCatalogs work has done. 
    However, doing so would cause highly non-linear impacts to our SFR plots after merger
    and to the stellar distribution plots post-merger. For now, we therefore
    use only no-delay models and note that realism can only be improved if 
    the underlying MBH merger algorithm improves. 

    THE CODE BELOW PRODUCES FAKE BINARIES PROPERTIES, 
    PLEASE REPLACE IT WITH CUSTOMARY CODE TO COLLECT BINARIES FROM YOUR MODEL.

    # Required fields
        N_binaries: number of binaries 
        
        #Black Hole Dictionary
        galid: Host Galaxy ID [None]
        m1: primary mass [M_sun]
        m2: secondary mass [M_sun]
        z: redshift of binary merger [None]
        sepa: separation at merger [proper kpc]
        W: number density [cMpc^-3] i.e. per comoving cubic Mpc 

        #Host Galaxy Dictionary
        galid: Host Galaxy ID [None] (sanity check. must match above ID)
        sfr{}: Star formation rate dictionary in galid galaxy for as long as we can get it up to delta_z = 5
        mstar: host galaxy stellar mass at merger or just after it [M_sun]
        zgal: redshift of galaxy stellar mass [None]
        R50: half-mass radius or effective radius [proper kpc]
        mhalo: Host halo mass at merger or just after [M_sun]
        galpos: Can be either central or satellite [string]
        metallicity: Metalicity of host galaxy at merger or just after [Z_msun]
    

    Returns: dictionary 
    '''
    ID = 0
    ###########################################################
    ####### CHANGE THE CODE BELOW #############################
    ###########################################################
    #*********************************************************#
    #*********************************************************#

    # generate fake binary properties
    N_binaries = 1000

    #Black Hole properties
    galid = np.arange(N_binaries)
    m1 = np.random.normal(loc=1e7, scale=1e6, size=N_binaries)
    m2 = np.random.normal(loc=1e7, scale=1e6, size=N_binaries)
    m1, m2 = np.max([m1, m2], axis=0), np.min([m1, m2], axis=0)
    z = 0.01 + np.random.uniform(0, 10, size=N_binaries)
    sepa = 10.0 ** np.random.uniform(2.0, 4.0, size=N_binaries)
    # number density
    W = 1/metadata["BoxSize"]**3*np.ones(len(m1))
    
    
    #Host galaxy properties

    # generate fake binary-host galaxy-remnant properties
    galid = galid
    sfr = {}
    mstar = np.random.normal(loc=1e10, scale=1e8, size=N_binaries)
    mdm = mstar * np.maximum(1.0, np.random.normal(loc=10.0, scale=1.0, size=N_binaries))
    zgal = z - np.random.uniform(0, 0.001, size=N_binaries)
    R50 = np.random.normal(loc=3, scale=0.1, size=N_binaries)
    metallicity =  np.random.normal(loc=1e10, scale=1e8, size=N_binaries)
    galpos = "central"
   
    # FILL METADATA INFO
    # total number of merged binaries assuming no delays
    metadata['NumberBinaries'] = N_binaries

    # provide an explanation of the merger criterion, modify the string
    metadata['MergerCriteria'] = (
        "mergers occur when two MBH particles come within a gravitational softening length of "
        "eachother, and the kinetic energy of the pair is less than the gravitational "
        "potential energy between them.")
    
    metadata['Comments'] = (
        "Any additional information you consider relevant for any clarification, i.e."
        "model special features, recipe to deal with MBH evolution etc.")

    #**********************************************************#
    #**********************************************************#
    ############################################################
    # DO NOT CHANGE DICTIONARY AND RETURN ######################
    ############################################################

    # collect data in a dictionary
    mbhenv = {
        "BlackHoles": {
            'GalaxyID': galid,
            'PrimaryMass': m1,
            'SecondaryMass': m2,
            'Redshift': z,
            'Separation': sepa,
            'NumberDensity': W,
        },
        "HostGalaxy": {
            'GalaxyID': galid,
            'SFR': sfr,
            'HostGalaxyStellarMass': mstar,
            'HostGalaxyHaloMass': mdm,
            'HostGalaxyRedshift': zgal,
            'HostGalaxyR50': R50,
            'HostGalaxyMetallicity': metallicity,
            'HostGalaxyPosition': galpos,
        }
    }

    return mbhenv

def write_catalog_hdf5(filename, metadata, mbhenv):
    """
    Write the MBH environment catalog to an HDF5 file.

    Parameters
    ----------
    filename : str
    metadata : dict
    mbhenv : dict with structure:
        mbhenv["BlackHoles"][field] = array
        mbhenv["HostGalaxy"][field] = array
    """

    with h5py.File(filename, "w") as f:

        # ----------------------------------------------------
        # Metadata group
        # ----------------------------------------------------
        gmeta = f.create_group("Metadata")
        for key, value in metadata.items():
            # store strings as fixed-length UTF-8
            if isinstance(value, str):
                gmeta.attrs[key] = np.string_(value)
            elif isinstance(value, list):
                # convert lists-of-strings to variable-length strings
                gmeta.attrs[key] = [np.string_(v) for v in value]
            else:
                gmeta.attrs[key] = value

        # ----------------------------------------------------
        # Black hole binary information
        # ----------------------------------------------------
        gbh = f.create_group("Binaries")
        for key, arr in mbhenv["BlackHoles"].items():
            gbh.create_dataset(key, data=np.array(arr),
                               compression="gzip")

        # ----------------------------------------------------
        # Host galaxy information
        # ----------------------------------------------------
        ggal = f.create_group("HostGalaxy")
        for key, arr in mbhenv["HostGalaxy"].items():
            # SFR is a special dictionary → store separately
            if key == "SFR":
                gsfr = ggal.create_group("SFR")
                for k2, arr2 in arr.items():
                    gsfr.create_dataset(k2, data=np.array(arr2),
                                        compression="gzip")
            else:
                ggal.create_dataset(key, data=np.array(arr),
                                    compression="gzip")

    print(f"[✓] Wrote catalog to {filename}")

def validate_catalog(filename):
    """
    Validate the HDF5 catalog written by write_catalog_hdf5().
    Ensures:
      - Required groups exist
      - Required datasets exist
      - Arrays have consistent lengths
      - No invalid dtypes (e.g. object)
      - No NaNs in required fields
    """

    required_groups = ["Metadata", "Binaries", "HostGalaxy"]

    required_bh_fields = [
        "GalaxyID",
        "PrimaryMass",
        "SecondaryMass",
        "Redshift",
        "Separation",
        "NumberDensity",
    ]

    required_gal_fields = [
        "GalaxyID",
        "HostGalaxyStellarMass",
        "HostGalaxyHaloMass",
        "HostGalaxyMetallicity",
        "HostGalaxyR50",
        "HostGalaxyRedshift",
        "HostGalaxyPosition",
    ]

    print("\n[Validator] Validating catalog:", filename)

    with h5py.File(filename, "r") as f:

        # ---- Check groups exist ----
        for g in required_groups:
            if g not in f:
                raise ValueError(f"[Validator] Missing group: {g}")
        print("[Validator] Groups OK")

        # ---- Check binary fields ----
        bh = f["Binaries"]
        for key in required_bh_fields:
            if key not in bh:
                raise ValueError(f"[Validator] Missing Binaries/{key}")
        print("[Validator] Binaries fields OK")

        # ---- Check host galaxy fields ----
        gal = f["HostGalaxy"]
        for key in required_gal_fields:
            if key not in gal:
                raise ValueError(f"[Validator] Missing HostGalaxy/{key}")
        print("[Validator] HostGalaxy fields OK")

        # ---- Check array lengths match ----
        N = len(bh["GalaxyID"])
        for key in required_bh_fields:
            if len(bh[key]) != N:
                raise ValueError(f"[Validator] Binaries/{key} wrong length")

        for key in required_gal_fields:
            if len(gal[key]) != N:
                raise ValueError(f"[Validator] HostGalaxy/{key} wrong length")
        print("[Validator] Field lengths OK (N = %d)" % N)

        # ---- Check no object dtypes ----
        for group, keys in [("Binaries", required_bh_fields),
                            ("HostGalaxy", required_gal_fields)]:
            for key in keys:
                arr = f[group][key][:]
                if arr.dtype == object:
                    raise TypeError(f"[Validator] {group}/{key} has object dtype")

        print("[Validator] Datatypes OK")

        # ---- Check no NaNs in required numeric fields ----
        numeric_bh_fields = ["PrimaryMass", "Redshift", "NumberDensity"]
        numeric_gal_fields = ["HostGalaxyStellarMass", "HostGalaxyHaloMass",
                              "HostGalaxyMetallicity", "HostGalaxyR50"]

        for key in numeric_bh_fields:
            if np.isnan(bh[key][:]).any():
                raise ValueError(f"[Validator] NaNs in Binaries/{key}")

        for key in numeric_gal_fields:
            if np.isnan(gal[key][:]).any():
                raise ValueError(f"[Validator] NaNs in HostGalaxy/{key}")

        print("[Validator] No NaNs in required fields")

    print("[Validator] ✓ Catalog validation PASSED\n")

def main():
    print("Generating MBH Environment Catalog...")

    metadata, mbhenv = input_data()
    filename = "MBH_Environment_Catalog_%s.hdf5" % (SIMULATION)
    write_catalog_hdf5(args.output, metadata, mbhenv)
    
    validate_catalog(filename)

if __name__ == "__main__":
    main()
