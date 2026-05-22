"""LISA AstroWG - MBH Environments - Catalog generation script - v1.0

This script used the script written by Matteo Bonetti & Luke Zoltan Kelley as a blueprint. 

Catalog v1.0 is primarily meant to collect data from datasets on
(i) the environments surrounding MBH (numerical) mergers


(i) information on binary MBHs for “no delay” cases that will be used to create a uniform “delay” case for all models, 
(ii) information on the full MBH population that will be used to build AGN luminosity functions and MBH-Mgal relations to compare to observations. 
A group can also provide a “delay” case that will be used for the paper, but this can also be provided at later time. 
There are some additional data, such as dark matter halo information, that can be optionally provided for v2.1 if desired.  

Links
-----
- Catalog v2.1 spreadsheet:
  https://docs.google.com/spreadsheets/d/1Zz-A5QcKHdPabpSmkKcs9ERzeAA0sqM_/edit?usp=sharing&ouid=113068423500393235248&rtpof=true&sd=true
- MBHCatalogs github for code:
  https://github.com/mbonetti90/MBHCatalogs
- MBHCatalogs google drive for simulation data:
  https://drive.google.com/drive/folders/1rW-cdOrMglfp2w72X0jmdu1G5COb7RDk?usp=sharing
- MBHCatalogs general folder
  https://drive.google.com/drive/folders/1YQYw-Km-N5b5jjeHYQ4Ls6LTbXxKh9VC?usp=sharing
Authors
-------
- Matteo Bonetti : matteo.bonetti@unimib.it
- Luke Zoltan Kelley : lzkelley@berkeley.edu

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

python3 hdf5_catalog_v2.1.py -W -F the_name_of_the_file_to_be_produced

type -h for immediate help


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


SIMULATION="SEEDZ"

def input_data():
    '''
    This function collects the necessary information to produce the MBH Env catalog.

    Users should perform the following actions:
    1) edit the 'metadata' dictionary in this function to provide with the specific information concerning a certain model;
    2) edit the 'MBHB_no_delay' function to provide with the no-delay binary catalog;
    3) [optional] edit the 'MBHB_delay' function to provide with the delay binary catalog. Leave an empty dict if the delay model is not present;
    4) edit the MBH_population function to provide with the complete MBH population at selected redshifts.

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

    Returns: 4 dictionaries -> metadata, allbhs, binaries_nodelay, binaries_delay
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
        'SimulationVersion': 'HighRes',
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
        'HubbleConstant': 67.4, #km s^-1 Mpc^-1
        'OmegaMatter': 0.315,
        'OmegaLambda': 0.685,
        'BoxSize': 1e0, # cMpc
        'MinBHSeedMass': 5e3, #M_sun
        'MinRedshift': 10,
        'MaxRedshift': 20,
        'StellarMassResolution': 1e3, # M_sun
        'DarkMatterMassResolution': 1e5, # M_sun
        'SpatialResolution': np.nan, # kpc
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
    """
    Load real MBH binary and host galaxy information from the Step 3
    output pickle and populate the MBH Environments catalog structure.
    """

    files = []
    files.append("Normal1_NoFB/galaxy_properties.pkl")
    files.append("Normal2_NoFB/galaxy_properties.pkl")
    files.append("Rarepeak_NoFB/galaxy_properties.pkl")

    print("Reading files: ", files)
    galaxyresults = {}
    for galfile in files:
        with open(galfile, "rb") as f:
            data = pickle.load(f)
        max_id = max(galaxyresults.keys()) if galaxyresults else 0
        for k, v in data.items():
            galaxyresults[max_id + k + 1] = v

        print(f"Loaded {galfile}: {len(data)} galaxies")
    galaxy_ids = sorted(galaxyresults.keys())
    N = len(galaxy_ids)
    print("N = ", N)
    # --- Preallocate arrays ---
    PrimaryMass        = np.zeros(N)
    SecondaryMass      = np.full(N, np.nan)   # Only primary BH mass available
    Redshift           = np.zeros(N)
    Separation         = np.full(N, np.nan)   # Fill with NaN (not provided)
    NumberDensity      = np.zeros(N)

    HostGalaxyStellarMass = np.zeros(N)
    HostGalaxyHaloMass    = np.zeros(N)
    HostGalaxyMetallicity = np.zeros(N)
    HostGalaxyR50         = np.zeros(N)
    HostGalaxyRedshift    = np.zeros(N)
    HostGalaxyPosition    = np.array(["central"] * N)

    Vbox = metadata["BoxSize"]**3

    for i, gid in enumerate(galaxy_ids):
        g = galaxyresults[gid]

        # Binary properties
        PrimaryMass[i]   = g["BHPrimaryMass"]
        SecondaryMass[i] = g["BHRemnantMass"]- g["BHPrimaryMass"]
        Redshift[i]      = g["Redshift"]
        NumberDensity[i] = 1.0 / Vbox

        # Host galaxy
        HostGalaxyStellarMass[i] = g["StellarMass"]
        HostGalaxyHaloMass[i]    = g["HaloMass"]
        HostGalaxyMetallicity[i] = g["GasMetallicity_MW"]
        HostGalaxyR50[i]         = g["R50_kpc"]
        HostGalaxyRedshift[i]    = g["Redshift"]

    metadata["NumberBinaries"] = N

    # --- Final mbhenv dictionary ---
    mbhenv = {
        "BlackHoles": {
            "GalaxyID": np.array(galaxy_ids),
            "PrimaryMass": PrimaryMass,
            "SecondaryMass": SecondaryMass,
            "Redshift": Redshift,
            "Separation": Separation,
            "NumberDensity": NumberDensity,
        },
        "HostGalaxy": {
            "GalaxyID": np.array(galaxy_ids),
            "HostGalaxyStellarMass": HostGalaxyStellarMass,
            "HostGalaxyHaloMass": HostGalaxyHaloMass,
            "HostGalaxyMetallicity": HostGalaxyMetallicity,
            "HostGalaxyR50": HostGalaxyR50,
            "HostGalaxyRedshift": HostGalaxyRedshift,
            "HostGalaxyPosition": HostGalaxyPosition,
            "SFR": {},  # optional
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
        # Host galaxy information (robust string handling)
        # ----------------------------------------------------
        ggal = f.create_group("HostGalaxy")
        for key, arr in mbhenv["HostGalaxy"].items():

            # SFR is a subgroup
            if key == "SFR":
                gsfr = ggal.create_group("SFR")
                for k2, arr2 in arr.items():
                    arr2_np = np.array(arr2)
                    gsfr.create_dataset(k2, data=arr2_np, compression="gzip")
                continue

            # Convert to numpy array *first*
            arr_np = np.array(arr)

            # Debug print (optional)
            # print("KEY:", key, "DTYPE:", arr_np.dtype, "EXAMPLE:", arr_np[0])
            
            # ---------- FIX 1: Replace None with np.nan ----------
            if arr_np.dtype == object:
                arr_np = np.array([np.nan if x is None else x for x in arr_np])
                
            # ---------- FIX 2: Convert remaining object strings ----------
            if arr_np.dtype == object:
                if all(isinstance(x, str) for x in arr_np):
                    arr_np = arr_np.astype('S')   # ASCII for HDF5
                else:
                    raise TypeError(f"Cannot store field '{key}' with mixed dtype object.")

            # ---------- FIX 3: Convert Unicode strings ----------
            if arr_np.dtype.kind == 'U':  # Unicode dtype <U...
                arr_np = arr_np.astype('S')

            # Create dataset
            ggal.create_dataset(key, data=arr_np, compression="gzip")

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
    import shutil
    print("Generating MBH Environment Catalog for %s..." % (SIMULATION))

    metadata, mbhenv = input_data()
  
    filename = "MBH_Environment_Catalog_%s.hdf5" % (SIMULATION)
    write_catalog_hdf5(filename, metadata, mbhenv)
    shutil.copy(filename, '../Catalogues/')
    print(f"[✓] Wrote final catalog to %s" % (filename))

    validate_catalog(filename)


if __name__ == "__main__":
    main()
