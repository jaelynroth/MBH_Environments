#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------
# Publication-quality Matplotlib styling
# ---------------------------------------------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 24,
    "lines.linewidth": 2,
    "axes.linewidth": 1.8,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# LOAD CATALOG
# ============================================================

def load_catalog(filename):
    with h5py.File(filename, "r") as f:

        # --- BH masses (from binaries) ---
        BH_Primary = np.array(f["Binaries"]["PrimaryMass"])
        BH_Secondary = np.array(f["Binaries"]["SecondaryMass"])
        # --- Galaxy properties ---
        Z = np.array(f["HostGalaxy"]["HostGalaxyMetallicity"])
        MStellar = np.array(f["HostGalaxy"]["HostGalaxyStellarMass"])
        Redshift = np.array(f["HostGalaxy"]["HostGalaxyRedshift"])

    return BH_Primary, BH_Secondary, Z, MStellar, Redshift


# ============================================================
# PLOTTING
# ============================================================

def make_plots(BH_Primary, BH_Secondary, Z, MStellar, Redshift):
    logging.info(f"Number of merger systems before masking: {len(BH_Primary)}")

    print("BH_Primary = ", BH_Primary)
    print("BH_Secondary = ", BH_Secondary)
    print("MStellar = ", MStellar)
    print("Z = ", Z)
    # ---- Mask invalid values ----
    mask = (BH_Primary > 0) & (BH_Secondary > 0) & (Z > 0) & (MStellar > 0) & \
       np.isfinite(BH_Primary) & np.isfinite(BH_Secondary) & np.isfinite(Z) & np.isfinite(MStellar)

    
    BH_Primary = BH_Primary[mask]
    BH_Secondary = BH_Secondary[mask]
    RemnantBHMass = BH_Primary + BH_Secondary
    Z  = Z[mask]
    MStellar = MStellar[mask]
    Redshift = Redshift[mask]

    
    logging.info(f"Number of merger systems after masking: {len(BH_Primary)}")
    print("Stellar Masses (min %e, max %e) " % (MStellar.min(), MStellar.max()))
    print("BH Masses (min %e, max %e) " % (RemnantBHMass.min(), RemnantBHMass.max()))
    print("Mass Weighted Metallicities (min %e, max %e) " % (Z.min(), Z.max()))
    # ---- Plot 1: BH Mass vs Metallicity ----
    plt.figure(figsize=(10,6))
    #plt.scatter(Z, RemnantBHMass, s=20, alpha=0.7, color="tab:blue")
    print(Z.min(), Z.max())
   
    plt.hexbin(Z, RemnantBHMass, gridsize=60, cmap='viridis',xscale='log', yscale='log', bins='log')
    plt.colorbar(label='Galaxies')
    #plt.xscale("log")
    #plt.yscale("log")
    #plt.xlim(1e-5, 1e0)
    #plt.clim(1, 50)

    plt.xlabel(r"Mass-Weighted Gas Metallicity [$\rm{Z/Z_\odot}$]")
    plt.ylabel(r"BH Remnant Mass [$\rm{M_\odot}$]")

    plt.tight_layout()
    plt.savefig("Catalogue_BHMass_Metallicity.png")

    # ---- Plot 2: BH Mass vs Stellar Mass ----
    plt.figure(figsize=(10,6))
    plt.hexbin(MStellar, RemnantBHMass, gridsize=60, cmap='viridis',xscale='log', yscale='log', bins='log')
    plt.colorbar(label='Galaxies')
    plt.xlabel(r"Stellar Mass [$\rm{M_\odot}$]")
    plt.ylabel(r"BH Remnant Mass [$\rm{M_\odot}$]")

    plt.tight_layout()
    plt.savefig("Catalogue_BHMass_StellarMass.png")

    # ---- Plot 3: BH Mass vs Merger Redshift ----
    plt.figure(figsize=(10,6))
    plt.hexbin(Redshift, RemnantBHMass,  gridsize=60, cmap='viridis', yscale='log', bins='log')
    plt.xlim(20, 10)
    plt.xlabel(r"Redshift")
    plt.ylabel(r"BH Remnant Mass [$\rm{M_\odot}$]")
    plt.colorbar(label='Galaxies')
    plt.tight_layout()
    plt.savefig("Catalogue_BHMass_Redshift.png")



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    filename = "/home/jregan/data/Analysis/ArepoYTAnalysis/MBHEnvironments/MBH_Environments/SEEDZ/Catalogues/MBH_Environment_Catalog_SEEDZ.hdf5"  # change to your actual file name

    print("Available Datasets:\n")
    with h5py.File(filename, 'r') as f:
        f.visit(print)
    print("\n")
    logging.info(f"Reading catalog: {filename}")
    BH_Primary, BH_Secondary, Z, MStellar, Redshift = load_catalog(filename)

    make_plots(BH_Primary, BH_Secondary, Z, MStellar, Redshift)
