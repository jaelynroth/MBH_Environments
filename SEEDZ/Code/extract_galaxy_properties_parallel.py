#!/usr/bin/env python3
import numpy as np
import yt
import pickle
import glob
import logging
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt


#Script Purpose
#loads all merger‑host galaxies from your Step 2 catalogue and, for each snapshot, extracts the gas, stellar, DM, metallicity, R50, and BH remnant mass for each merger event using yt.
#runs these extractions in parallel (one task per snapshot) and writes a pickle file containing a dictionary GalaxyID → {environment properties}.
#This pickle is the final structured dataset used to generate science plots — no rerunning of the yt extraction is required unless you want to recompute the physical quantities.

# ============================================================
# CONFIGURATION
# ============================================================
R_PROP_KPC = 1.0              # physical radius for environment sampling
DEDUP_RADIUS_KPC = 1.0
MIN_STARS = 5

# Plot styling (matching plot_mergers.py)
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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)

# ============================================================
# UTILITIES
# ============================================================

def compute_R50(center_code, pos_code, mass, ds):
    """Stellar half-mass radius (kpc)."""
    if len(pos_code) < MIN_STARS:
        return np.nan
    center = ds.arr(center_code, "code_length")
    pos    = ds.arr(pos_code, "code_length")
    dist   = (pos - center).to("kpc").v
    dist   = np.linalg.norm(dist, axis=1)
    idx    = np.argsort(dist)
    dist_s = dist[idx]
    mass_s = mass[idx]
    cum    = np.cumsum(mass_s)
    return float(dist_s[np.searchsorted(cum, cum[-1]/2)])


def compute_metallicity(g_sne, g_mass):
    """Metallicity in Z/Zsun using SneTracerField."""
    Z = np.divide(g_sne, g_mass,
                  out=np.zeros_like(g_sne),
                  where=g_mass > 0)
    Z /= 0.02
    Z[Z < 1e-20] = 1e-20
    return Z


def deduplicate_galaxies(glist, ds):
    """Merge galaxies whose BH centers are <1 kpc apart."""
    unique = []
    for g in glist:
        c_code = np.array(g["Center"])  # comoving code_length coordinates
        c_kpc  = ds.arr(c_code, "code_length").to("kpc").v
        matched = False
        for u in unique:
            if np.linalg.norm(c_kpc - u["center_kpc"]) < DEDUP_RADIUS_KPC:
                u["galaxy_ids"].append(g["GalaxyID"])
                u["bh_ids"].append(g["PrimaryID"])
                matched = True
                break
        if not matched:
            unique.append({
                "center_code": c_code,
                "center_kpc" : c_kpc,
                "snapshot": g["Snapshot"],
                "redshift": g["Redshift"],
                "galaxy_ids": [g["GalaxyID"]],
                "bh_ids": [g["PrimaryID"]],
            })
    return unique

# ============================================================
# PROCESS ONE SNAPSHOT
# ============================================================

def process_snapshot(args):
    snap, galaxies, snap_base, sinks = args

    logging.info(f"[Snapshot {snap}] Loading snapshot...")
    ds = yt.load(f"{snap_base}/snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5")
    ad = ds.all_data()
    h = ds.hubble_constant

    logging.info(f"[Snapshot {snap}] Preloading particle fields...")

    # Gas
    gas_pos  = ad[("PartType0","Coordinates")].v       # code_length
    gas_mass = ad[("PartType0","Masses")].to("Msun").v
    gas_sne  = ad[("PartType0","SneTracerField")].v

    # DM
    try:
        dm_pos  = ad[("PartType1","Coordinates")].v
        dm_mass = ad[("PartType1","Masses")].to("Msun").v
    except:
        dm_pos  = np.zeros((0,3))
        dm_mass = np.zeros(0)

    # Stars (Type 4) (inactive PopII stars)
    star4_pos  = ad[("PartType4","Coordinates")].v
    star4_mass = ad[("PartType4","Masses")].to("Msun").v

    # PartType5 (active PopII, PopIII, BHs)
    p5_pos  = ad[("PartType5","Coordinates")].v
    p5_mass = ad[("PartType5","Masses")].to("Msun").v
    p5_ids  = ad[("PartType5","ParticleIDs")].v.astype(int)

    # Deduplicate galaxies
    unique = deduplicate_galaxies(galaxies, ds)
    logging.info(f"[Snapshot {snap}] Unique galaxies: {len(unique)}")

    # classify PartType5 into stars vs BH
    is_star = []
    for pid in p5_ids:
        if pid in sinks:
            t = sinks[pid]["meta"]["Type"]
            is_star.append(t not in ("MBH", "BH", 3))
        else:
            is_star.append(True)
    is_star = np.array(is_star)

    # combined stars
    star_pos  = np.vstack([star4_pos, p5_pos[is_star]])
    star_mass = np.concatenate([star4_mass, p5_mass[is_star]])

    results = {}

    for idx, g in enumerate(unique):

        galID = min(g["galaxy_ids"])
        z     = g["redshift"]
        center_code = g["center_code"]   # comoving ckpc/h

        # ---- region definition ----
        center_yt = ds.arr(center_code, "code_length")
        region    = ds.sphere(center_yt, (R_PROP_KPC, "kpc"))

        Mgas = float(region[("PartType0","Masses")].to("Msun").sum())

        if Mgas > 0:
            sne   = region[("PartType0","SneTracerField")].v
            gmass = region[("PartType0","Masses")].v

            Zarr  = compute_metallicity(sne, gmass)
            Zmw   = float((Zarr * gmass).sum() / gmass.sum())
            Zmin  = float(np.nanmin(Zarr))
            Zmax  = float(np.nanmax(Zarr))
        else:
            Zmw = Zmin = Zmax = np.nan

        if ("PartType1","Masses") in ds.field_list:
            try:
                Mdm = float(region[("PartType1","Masses")].to("Msun").sum())
            except yt.utilities.exceptions.YTFieldNotFound:
                Mdm = np.nan
        else:
            Mdm = np.nan

        # ---- stars ----
        Mstar = float(region[("PartType4","Masses")].to("Msun").sum())
        R50   = compute_R50(center_code, star_pos, star_mass, ds)

        # ---- BH MASS ----
       
        BH_id = g["bh_ids"][0]

        # Skip if BH does not exist in sink evolution
        if BH_id not in sinks:
            logging.warning(f"[Snapshot {snap}] BH id {BH_id} not found in sinks. Skipping galaxy {galID}.")
            continue

        a_snap = 1/(1+z)

        tvals = np.array(list(sinks[BH_id]["evolution"].keys()))

        t_sel = tvals[np.argmin(np.abs(tvals - a_snap))]
        evo   = sinks[BH_id]["evolution"][t_sel]

        ptype = sinks[BH_id]["meta"]["Type"]
        if ptype not in ("MBH","BH",3):
            logging.warning(f"BH id {BH_id} has Type={ptype}, expected MBH")
            BH_PrimaryMass = np.nan
            BH_RemnantMass = np.nan
        else:
            BH_PrimaryMass_code = evo["StellarMass"]
            BH_PrimaryMass = BH_PrimaryMass_code * (1e10 / h)     # convert to Msun
            print("BH Primary Mass = %e Msun" % ( BH_PrimaryMass))
            BH_RemnantMass_code = evo["MergerMass"] + BH_PrimaryMass_code
            BH_RemnantMass = BH_RemnantMass_code * (1e10 / h)     # convert to Msun
            print("BH Remnant Mass = %e Msun" % ( BH_RemnantMass))
            if(BH_PrimaryMass > BH_RemnantMass):
                print("!!!!!Failure. The Primary mass is greater than the remnant mass. How can this be?!!!!")
                sys.exit()

        # ---- DEBUG (first 5 galaxies per snapshot) ----
        if idx < 5:
            print("\n------ DEBUG SNAPSHOT", snap, " GALAXY", galID, " ------")
            print("z =", z)
            print("BH id:", BH_id)
            print("BH Type:", ptype)
            print("Sink position (code units):", evo["Pos"])
            print("Center (code units):", center_code)
            print("Center (kpc):", center_yt.to("kpc").v)
            print("BH Remnant Mass (Msun):", BH_RemnantMass)
            print("Metallicity mass-weighted:", Zmw)
            print("Metallicity min:", Zmin)
            print("Metallicity max:", Zmax)
            print("Gas mass:", Mgas)

        # ---- Save ----
        results[galID] = {
            "GalaxyID": galID,
            "Snapshot": snap,
            "Redshift": z,
            "Center_code": center_code.tolist(),
            "BHRemnantMass": BH_RemnantMass,
            "BHPrimaryMass": BH_PrimaryMass,
            "GasMass": Mgas,
            "HaloMass": Mdm,
            "StellarMass": Mstar,
            "GasMetallicity_MW": Zmw,
            "GasMetallicity_Min": Zmin,
            "GasMetallicity_Max": Zmax,
            "R50_kpc": R50,
        }

    return results

# ============================================================
# MASTER FUNCTION
# ============================================================

def extract_galaxy_properties(
        snap_base, base, 
        galaxies_file="merger_galaxies.pkl",
        sinks_file="sink_particle.pkl",
        outfile="galaxy_properties.pkl"
    ):

    logging.info("Loading sinks...")
    from DataReader import Reader
    R = Reader(base)
    sinks = R.pickle_reader(sinks_file)["data"]

    # ------------------------------------------------------------
    # Diagnostics: Inspect sink contents
    # ------------------------------------------------------------
    
    print("\n=== SINK SUMMARY ===")
    
    # 1. Number of sinks
    num_sinks = len(sinks)
    print(f"Total number of sinks: {num_sinks}")
    
    # 2. Types of sinks (PopII, PopIII, MBH)
    from collections import Counter
    sink_types = Counter(sd["meta"]["Type"] for sd in sinks.values())
    print("Sink types and counts:", sink_types)
    
    # 3. MBH sinks — count how many survive to z ≈ 10 (a ~ 0.0909)
    #    A sink "exists" at z=10 if it has an evolution entry at a >= 0.0908
    a_z10 = 1.0 / (1.0 + 10.0)   # = 0.090909...
    MBHs_at_z10 = [
        sid for sid, sd in sinks.items()
        if sd["meta"]["Type"] == "MBH" and
        any(t >= 0.0908 for t in sd["evolution"].keys())
    ]
    print(f"Number of MBH sinks that exist at z=10: {len(MBHs_at_z10)}")
    
    # 4. Mass of each MBH at z = 10 (converted to Msun)
    #    Note: StellarMass is in code units 1e10 Msun/h
    h = 0.67
    MBH_masses = []
    
    for sid, sd in sinks.items():
        if sd["meta"]["Type"] != "MBH":
            continue
        
        # find the evolution entry closest to a = 0.0909
        evo = sd["evolution"]
        times = sorted(evo.keys(), key=lambda x: abs(x - a_z10))
        t_closest = times[0]
        
        mass_code = evo[t_closest]["StellarMass"]
        mass_msun = mass_code * 1e10 / h
        
        MBH_masses.append(mass_msun)
        
    if MBH_masses:
        print(f"Most massive MBH mass at z=10: {max(MBH_masses):.3e} Msun")
    else:
        print("No MBH masses could be extracted at z=10.")

    print("=== END SINK SUMMARY ===\n")


    #Moving onto Galaxies analysis
    logging.info("Loading galaxy list...")
    with open(galaxies_file, "rb") as f:
        galaxies = pickle.load(f)

    # One task per merger, processed at the correct snapshot
    tasks = [(g["Snapshot"], [g], snap_base, sinks) for g in galaxies]
    
    workers = min(32, len(tasks))
    logging.info(f"Using {workers} workers for {len(tasks)} mergers")
    
    with Pool(workers) as pool:
        results_list = pool.map(process_snapshot, tasks)

    results = {}
    for r in results_list:
        results.update(r)

    # Save
    with open(outfile, "wb") as f:
        pickle.dump(results, f)

    logging.info(f"Saved {len(results)} galaxies → {outfile}")

    # ---- Plots ----
    BH = np.array([v["BHPrimaryMass"] for v in results.values()])
    Z  = np.array([v["GasMetallicity_MW"] for v in results.values()])
    MStellar =  np.array([v["StellarMass"] for v in results.values()])

    mask = (BH > 0) & (Z > 0) & np.isfinite(BH) & np.isfinite(Z)
    BH = BH[mask]
    Z  = Z[mask]
    MStellar = MStellar[mask]
    
    plt.figure(figsize=(8,6))
    plt.scatter(Z, BH, s=20, alpha=0.7, color="tab:blue")
    plt.xlim(1e-5, 1e0)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Mass-Weighted Gas Metallicity [$Z/Z_\odot$]")
    plt.ylabel(r"BH Primary Mass [$M_\odot$]")
    plt.tight_layout()
    plt.savefig("BHMass_vs_Metallicity.png")
    logging.info("Saved BHMass_vs_Metallicity.png")


    plt.figure(figsize=(8,6))
    plt.scatter(MStellar, BH, s=20, alpha=0.7, color="tab:blue")
    plt.xlim(1e3, 1e10)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Stellar Mass [$M_\odot$]")
    plt.ylabel(r"BH Primary Mass [$M_\odot$]")
    plt.tight_layout()
    plt.savefig("BHMass_vs_StellarMass.png")
    logging.info("Saved BHMass_vs_StellarMass.png")

    return results

# ============================================================
if __name__ == "__main__":
    snap_base = "/home/daxal/data/ProductionRuns/Renaissance/NoFeedback/"
    base = "./test/"
    extract_galaxy_properties(snap_base, base)
