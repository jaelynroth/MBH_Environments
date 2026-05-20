#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

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

    return BH_Primary, BH_Secondary, Z, MStellar


# ============================================================
# PLOTTING
# ============================================================

def make_plots(BH_Primary, BH_Secondary, Z, MStellar):

    # ---- Mask invalid values ----
    mask = (BH_Primary > 0) & (BH_Secondary > 0) & (Z > 0) & (MStellar > 0) & \
       np.isfinite(BH_Primary) & np.isfinite(BH_Secondary) & np.isfinite(Z) & np.isfinite(MStellar)

    mask = (BH_Primary > 0) & (Z > 0) & \
       np.isfinite(BH_Primary) & np.isfinite(Z)


    
    RemnantBHMass = BH_Primary[mask] + BH_Secondary[mask]
    Z  = Z[mask]
    MStellar = MStellar[mask]

    logging.info(f"Number of valid systems: {len(BH_Primary)}")

    # ---- Plot 1: BH Mass vs Metallicity ----
    plt.figure(figsize=(8,6))
    plt.scatter(Z, RemnantBHMass, s=20, alpha=0.7, color="tab:blue")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1e-5, 1e0)

    plt.xlabel(r"Mass-Weighted Gas Metallicity [$Z/Z_\odot$]")
    plt.ylabel(r"BH Remnant Mass [$M_\odot$]")

    plt.tight_layout()
    plt.savefig("Catalogue_BHMass_Metallicity.png")
    logging.info("Saved BHMass_vs_Metallicity.png")

    # ---- Plot 2: BH Mass vs Stellar Mass ----
    plt.figure(figsize=(8,6))
    plt.scatter(MStellar, RemnantBHMass, s=20, alpha=0.7, color="tab:blue")
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r"Stellar Mass [$M_\odot$]")
    plt.ylabel(r"BH Remnant Mass [$M_\odot$]")

    plt.tight_layout()
    plt.savefig("Catalogue_BHMass_StellarMass.png")
    logging.info("Saved BHMass_vs_StellarMass.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    filename = "SEEDZ/Catalogues/MBH_Environment_Catalog_SEEDZ.hdf5"  # change to your actual file name

    logging.info(f"Reading catalog: {filename}")
    BH_Primary, BH_Secondary, Z, MStellar = load_catalog(filename)

    make_plots(BH_Primary, BH_Secondary, Z, MStellar)
