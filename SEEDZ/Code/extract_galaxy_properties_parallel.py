#!/usr/bin/env python3
import numpy as np
import yt
import h5py
import pickle
import glob
import logging
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt


#Script Purpose
#loads all merger‑host galaxies from your Step 2 catalogue and, for each snapshot, extracts the gas, stellar, DM, metallicity, R50, and BH remnant mass for each merger event using yt.
#runs these extractions in parallel (one task per snapshot) and writes a pickle file containing a dictionary GalaxyID → {environment properties}.
#This pickle is the final structured dataset used to generate science plots — no rerunning of the yt extraction is required unless you want to recompute the physical quantities.
#Now also extracts Subfind halo properties (M200, R200, SubhaloMassType) for each merger event.

# ============================================================
# CONFIGURATION
# ============================================================
R_PROP_KPC = 1.0              # physical radius for environment sampling (aperture)
DEDUP_RADIUS_KPC = 1.0
MIN_STARS = 5
MAX_MATCH_DIST_KPC = 10.0     # maximum distance for BH-subhalo matching
DEBUG = False                  # serial mode with limited tasks for debugging; set False for production
DEBUG_NTASKS = 5              # number of tasks to process in debug mode

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
                u["merger_times"].append(g["MergerTime"])
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
                "merger_times": [g["MergerTime"]],
            })
    return unique


# ============================================================
# SUBFIND UTILITIES
# ============================================================

def load_subfind(snap, snap_base):
    """
    Load and concatenate all Subfind FOF/subhalo files for a given snapshot.
    Converts positions and radii to physical kpc, masses to Msun.
    snap_base: base directory containing both snapdir_NNN and groups_NNN directories.
    """
    pattern = f"{snap_base}/groups_{snap:03d}/fof_subhalo_tab_{snap:03d}.*.hdf5"
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split(".")[-2]))

    if not files:
        logging.warning(f"[Subfind snap {snap}] No subfind files found at {pattern}")
        return None

    with h5py.File(files[0], "r") as f:
        header = dict(f["Header"].attrs)

    h = float(header.get("HubbleParam", 0.674))
    z = float(header["Redshift"])
    a = 1.0 / (1.0 + z)

    subhalo_pos, subhalo_mass_type, subhalo_halfrad, subhalo_group_nr = [], [], [], []
    group_r200, group_m200 = [], []

    for fname in files:
        with h5py.File(fname, "r") as f:
            if f["Header"].attrs["Nsubhalos_ThisFile"] > 0:
                subhalo_pos.append(f["Subhalo/SubhaloPos"][:])
                subhalo_mass_type.append(f["Subhalo/SubhaloMassType"][:])
                subhalo_halfrad.append(f["Subhalo/SubhaloHalfmassRadType"][:])
                subhalo_group_nr.append(f["Subhalo/SubhaloGroupNr"][:])
            if f["Header"].attrs["Ngroups_ThisFile"] > 0:
                group_r200.append(f["Group/Group_R_Crit200"][:])
                group_m200.append(f["Group/Group_M_Crit200"][:])

    if not subhalo_pos:
        logging.warning(f"[Subfind snap {snap}] No subhalos found in any file")
        return None

    subhalo_pos       = np.concatenate(subhalo_pos, axis=0)
    subhalo_mass_type = np.concatenate(subhalo_mass_type, axis=0)
    subhalo_halfrad   = np.concatenate(subhalo_halfrad, axis=0)
    subhalo_group_nr  = np.concatenate(subhalo_group_nr, axis=0).astype(int)
    group_r200        = np.concatenate(group_r200, axis=0)
    group_m200        = np.concatenate(group_m200, axis=0)

    # Positions and radii kept in raw comoving kpc/h for matching
    # Masses converted: code units (1e10 Msun/h) → Msun
    subhalo_mass_msun = subhalo_mass_type * 1e10 / h
    group_m200_msun   = group_m200 * 1e10 / h

    logging.info(f"[Subfind snap {snap}] Loaded {len(subhalo_pos)} subhalos, {len(group_r200)} groups")

    return {
        "subhalo_pos":       subhalo_pos,             # (N, 3) comoving kpc/h — for matching
        "subhalo_mass_type": subhalo_mass_msun,       # (N, 6) Msun — index 0=gas,1=DM,4=stars,5=sinks
        "subhalo_halfrad":   subhalo_halfrad,         # (N, 6) comoving kpc/h — converted to physical on output
        "subhalo_group_nr":  subhalo_group_nr,        # (N,) index into group arrays
        "group_r200":        group_r200,              # (M,) comoving kpc/h — converted to physical on output
        "group_m200":        group_m200_msun,         # (M,) Msun
        "h": h, "z": z, "a": a,
    }


def match_bh_to_subhalo(bh_pos_code, ds, subfind):
    """
    Match a BH position to the nearest subhalo.
    All positions are in comoving kpc/h for consistent matching.
    Radii converted to physical kpc on output.
    Returns match dict or None if no subhalo within MAX_MATCH_DIST_KPC (physical kpc).
    Note: SubhaloMassType[:,5] includes both BH sinks and stellar sinks —
    use existing is_star decontamination for clean stellar mass from Type5.
    """
    h = subfind["h"]
    a = subfind["a"]

    # BH position is already in comoving kpc/h (code_length units)
    bh_pos_raw = np.array(bh_pos_code)  # comoving kpc/h

    # Subfind positions are also in comoving kpc/h — match in same units
    dists_comoving = np.linalg.norm(subfind["subhalo_pos"] - bh_pos_raw, axis=1)
    idx = np.argmin(dists_comoving)

    # Convert match distance to physical kpc for threshold check and output
    min_dist_phys = float(dists_comoving[idx]) * a / h

    if min_dist_phys > MAX_MATCH_DIST_KPC:
        logging.warning(f"BH match distance {min_dist_phys:.2f} kpc exceeds {MAX_MATCH_DIST_KPC} kpc threshold — no host subhalo")
        return None

    group_idx = int(subfind["subhalo_group_nr"][idx])

    # Convert radii to physical kpc for output
    r200_phys     = float(subfind["group_r200"][group_idx]) * a / h
    halfrad_phys  = subfind["subhalo_halfrad"][idx] * a / h  # (6,) physical kpc

    return {
        "subhalo_idx":     idx,
        "match_dist_kpc":  min_dist_phys,
        "group_idx":       group_idx,
        "R200_kpc":        r200_phys,
        "M200_Msun":       float(subfind["group_m200"][group_idx]),
        "SubhaloMassType": subfind["subhalo_mass_type"][idx],   # (6,) Msun, all types
        "SubhaloHalfRad":  halfrad_phys,                        # (6,) physical kpc
    }


# ============================================================
# PROCESS ONE SNAPSHOT
# ============================================================

def process_snapshot(args):
    snap, galaxies, snap_base, sinks, mergers_by_id = args

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

    # combined stars (Type4 + Type5 stars only)
    star_pos  = np.vstack([star4_pos, p5_pos[is_star]])
    star_mass = np.concatenate([star4_mass, p5_mass[is_star]])

    # ---- Load Subfind for this snapshot ----
    subfind = load_subfind(snap, snap_base)
    if subfind is None:
        logging.warning(f"[Snapshot {snap}] No Subfind data available — Subfind quantities will be NaN")

    results = {}

    for idx, g in enumerate(unique):

        galID = min(g["galaxy_ids"])
        z     = g["redshift"]
        center_code = g["center_code"]   # comoving ckpc/h

        # ---- 1 kpc aperture region (keep for comparison) ----
        center_yt = ds.arr(center_code, "code_length")
        region    = ds.sphere(center_yt, (R_PROP_KPC, "kpc"))

        # ---- Aperture gas properties ----
        Mgas_ap = float(region[("PartType0","Masses")].to("Msun").sum())

        if Mgas_ap > 0:
            sne   = region[("PartType0","SneTracerField")].v
            gmass = region[("PartType0","Masses")].v
            Zarr  = compute_metallicity(sne, gmass)
            Zmw   = float((Zarr * gmass).sum() / gmass.sum())
            Zmin  = float(np.nanmin(Zarr))
            Zmax  = float(np.nanmax(Zarr))
        else:
            Zmw = Zmin = Zmax = np.nan

        # ---- Aperture DM ----
        if ("PartType1","Masses") in ds.field_list:
            try:
                Mdm_ap = float(region[("PartType1","Masses")].to("Msun").sum())
            except yt.utilities.exceptions.YTFieldNotFound:
                Mdm_ap = np.nan
        else:
            Mdm_ap = np.nan

        # ---- Aperture stellar mass (Type4 only — conservative) ----
        Mstar_ap = float(region[("PartType4","Masses")].to("Msun").sum())

        # ---- Aperture R50 ----

        # Get stars within aperture region only
        region_star4_pos  = np.asarray(region[("PartType4", "Coordinates")].v)
        region_star4_mass = np.asarray(region[("PartType4", "Masses")].to("Msun").v, dtype=float)
        
        region_p5_pos     = np.asarray(region[("PartType5", "Coordinates")].v)
        region_p5_mass    = np.asarray(region[("PartType5", "Masses")].to("Msun").v, dtype=float)
        region_p5_ids     = np.asarray(region[("PartType5", "ParticleIDs")].v, dtype=int)
        
        is_star_region = np.array([
            sinks[pid]["meta"]["Type"] not in ("MBH", "BH", 3)
            if pid in sinks else True
            for pid in region_p5_ids
        ], dtype=bool)
        
        # Safety: ensure mask is a proper 1D boolean array
        is_star_region = np.atleast_1d(np.asarray(is_star_region, dtype=bool))
        region_p5_mass = np.atleast_1d(region_p5_mass)
        region_p5_pos  = np.atleast_2d(region_p5_pos)
        
        # Optional sanity check
        if region_p5_mass.shape[0] != is_star_region.shape[0]:
            raise ValueError(
                f"Mask length mismatch: region_p5_mass={region_p5_mass.shape[0]}, "
                f"is_star_region={is_star_region.shape[0]}"
            )

        # Build combined stellar arrays safely
        if region_p5_mass.size > 0 and np.any(is_star_region):
            region_star_mass = np.concatenate([
                region_star4_mass,
                region_p5_mass[is_star_region]
            ])
            
            region_star_pos = np.vstack([
                region_star4_pos,
                region_p5_pos[is_star_region]
            ]) if region_star4_pos.size > 0 else region_p5_pos[is_star_region]
        else:
            region_star_mass = region_star4_mass
            region_star_pos  = region_star4_pos
            
        R50_ap = compute_R50(center_code, region_star_pos, region_star_mass, ds)

        # ---- BH MASS ----
        BH_id = g["bh_ids"][0]
        a_snap = 1/(1+z)
        
        if BH_id not in sinks:
            logging.warning(f"[Snapshot {snap}] BH id {BH_id} not found in sinks. Skipping galaxy {galID}.")
            continue

        print("Extracting BH massses.....")
            
        # Then per BH, look up the matching merger entry
        bh_mergers = mergers_by_id[BH_id]
        matched_merger = next((m for m in bh_mergers if abs(m["Time"] - g["merger_times"][0]) < 1e-6), None)
        assert matched_merger is not None, \
            f"BH {BH_id}: no merger found with Time={g['merger_times']} — upstream data error"
        
        BH_PrimaryMass      = matched_merger["M1"]
        BH_RemnantMass      = matched_merger["M1"] + matched_merger["M2"]
        print("M2 (secondary) = %e Msun  M1 (primary) = %e Msun" % (matched_merger["M2"], BH_PrimaryMass))
        # ---- Subfind matching ----
        M200         = np.nan
        R200_kpc     = np.nan
        Mstar_sub    = np.nan  # Type4 + decontaminated Type5 stellar mass within R200
        Mgas_sub     = np.nan  # SubhaloMassType[:,0]
        Mdm_sub      = np.nan  # SubhaloMassType[:,1]
        R50_sub      = np.nan  # SubhaloHalfmassRadType[:,4]
        match_dist   = np.nan
        subfind_matched = False

        if subfind is not None:
            match = match_bh_to_subhalo(center_code, ds, subfind)
            if match is not None:
                subfind_matched = True
                match_dist   = match["match_dist_kpc"]
                M200         = match["M200_Msun"]
                R200_kpc     = match["R200_kpc"]
                Mgas_sub     = float(match["SubhaloMassType"][0])
                Mdm_sub      = float(match["SubhaloMassType"][1])
                R50_sub      = float(match["SubhaloHalfRad"][4])  # stellar half-mass rad (Type4)

                # Stellar mass = SubhaloMassType[:,4] (dormant Type4)
                #              + decontaminated Type5 stellar mass within R200
                # SubhaloMassType[:,5] lumps BH sinks + stellar sinks so we can't use it directly
                Mstar4_sub = float(match["SubhaloMassType"][4])
                try:
                    region_r200  = ds.sphere(center_yt, (R200_kpc, "kpc"))
                    p5_ids_r200  = region_r200[("PartType5","ParticleIDs")].v.astype(int)
                    p5_mass_r200 = region_r200[("PartType5","Masses")].to("Msun").v
                    is_star_r200 = np.array([
                        sinks[pid]["meta"]["Type"] not in ("MBH","BH",3)
                        if pid in sinks else True
                        for pid in p5_ids_r200
                    ], dtype=bool)
                    Mstar5_sub = float(p5_mass_r200[is_star_r200].sum())
                except Exception as e:
                    logging.warning(f"[Snapshot {snap}] Type5 stellar mass within R200 failed: {e}")
                    Mstar5_sub = 0.0

                Mstar_sub = Mstar4_sub + Mstar5_sub

        # ---- DEBUG (first 5 galaxies per snapshot) ----
        if idx < 5:
            print("\n------ DEBUG SNAPSHOT", snap, " GALAXY", galID, " ------")
            print("z =", z)
            print("BH id:", BH_id)
            print("BH Type:", sinks[BH_id]["meta"]["Type"] if BH_id in sinks else "unknown")
            print("BH Remnant Mass (Msun): %e" % (BH_RemnantMass))
            print("BH Primary Mass (Msun): %e" % (BH_PrimaryMass))
            #print("BH Secondary Mass (Msun): %e" % (BH_SecondaryMass))
            print(f"--- Aperture ({R_PROP_KPC} kpc) ---")
            print("Gas mass: %e" % (Mgas_ap))
            # Type5 stellar mass within 1 kpc aperture
            region_p5_ids  = region[("PartType5","ParticleIDs")].v.astype(int)
            region_p5_mass = region[("PartType5","Masses")].to("Msun").v
            is_star_ap = np.array([
                sinks[pid]["meta"]["Type"] not in ("MBH","BH",3)
                if pid in sinks else True
                for pid in region_p5_ids
            ], dtype=bool)
            Mstar5_ap = float(region_p5_mass[is_star_ap].sum())
            print(f"Stellar mass (Type4): {Mstar_ap:.3e}  Type5_stars: {Mstar5_ap:.3e}  Total: {Mstar_ap + Mstar5_ap:.3e}")
            print("Metallicity mass-weighted:", Zmw)
            print("R50 (kpc):", R50_ap)
            print("--- Subfind ---")
            print("Matched:", subfind_matched, f"(dist={match_dist:.2f} kpc)" if subfind_matched else "")
            print("M200 (Msun): %e" % (M200))
            print("R200 (kpc):", R200_kpc)
            print("Subfind Mstar (Msun):", Mstar_sub, f"(Type4={Mstar4_sub:.3e}, Type5_stars={Mstar5_sub:.3e})" if subfind_matched else "")
            print("Subfind Mgas (Msun): %e" % (Mgas_sub))
            print("Subfind R50 (kpc):", R50_sub)




        #Need to decide what to write to the results
        print("match_dist = %f kpc\t" % (match_dist), "subfind_matched = ", subfind_matched)
       
        if((match_dist > 0.5) or (subfind_matched==False)):  #somewhat arbitrary but reasonable for high-z for cases where subfind clearly failing
            print("Using Aperature values")
            Mstar = Mstar_ap
            Mgas = Mgas_ap
            Mdm = Mdm_ap
            R50 = R50_ap
        else:
            print("Using subfind values")
            Mstar = Mstar_sub
            Mgas = Mgas_sub
            Mdm = Mdm_sub
            R50 = R50_sub
           
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
    merger_file = base+"mergers.pkl"
    galaxies_file = base+galaxies_file
    with open(merger_file, "rb") as f:
        mergers = pickle.load(f)
    from collections import defaultdict
    mergers_by_id = defaultdict(list)
    for m in mergers:
        mergers_by_id[m["SinkID"]].append(m)
    # ------------------------------------------------------------
    # Diagnostics: Inspect sink contents
    # ------------------------------------------------------------
    print("\n=== SINK SUMMARY ===")
    num_sinks = len(sinks)
    print(f"Total number of sinks: {num_sinks}")

    from collections import Counter
    sink_types = Counter(sd["meta"]["Type"] for sd in sinks.values())
    print("Sink types and counts:", sink_types)

    a_z10 = 1.0 / (1.0 + 10.0)
    MBHs_at_z10 = [
        sid for sid, sd in sinks.items()
        if sd["meta"]["Type"] == "MBH" and
        any(t >= 0.0908 for t in sd["evolution"].keys())
    ]
    print(f"Number of MBH sinks that exist at z=10: {len(MBHs_at_z10)}")

    h = 0.67
    MBH_masses = []
    for sid, sd in sinks.items():
        if sd["meta"]["Type"] != "MBH":
            continue
        evo = sd["evolution"]
        times = sorted(evo.keys(), key=lambda x: abs(x - a_z10))
        t_closest = times[0]
        mass_code = evo[t_closest]["StellarMass"]
        MBH_masses.append(mass_code * 1e10 / h)

    if MBH_masses:
        print(f"Most massive MBH mass at z=10: {max(MBH_masses):.3e} Msun")
    else:
        print("No MBH masses could be extracted at z=10.")
    print("=== END SINK SUMMARY ===\n")

    # ------------------------------------------------------------
    logging.info("Loading galaxy list...")
    with open(galaxies_file, "rb") as f:
        galaxies = pickle.load(f)

    # One task per merger, processed at the correct snapshot
    tasks = [(g["Snapshot"], [g], snap_base, sinks, mergers_by_id) for g in galaxies]

    if DEBUG:
        logging.info(f"DEBUG MODE: running serially on first {DEBUG_NTASKS} tasks")
        results_list = [process_snapshot(task) for task in tasks[:DEBUG_NTASKS]]
    else:
        workers = min(32, len(tasks))
        logging.info(f"Using {workers} workers for {len(tasks)} mergers")
        with Pool(workers) as pool:
            results_list = pool.map(process_snapshot, tasks)

    results = {}
    for r in results_list:
        results.update(r)

    with open(outfile, "wb") as f:
        pickle.dump(results, f)

    logging.info(f"Saved {len(results)} galaxies → {outfile}")

    # ---- Plots ----
    BH       = np.array([v["BHPrimaryMass"] for v in results.values()])
    Z        = np.array([v["GasMetallicity_MW"] for v in results.values()])
    Mstar = np.array([v["StellarMass"] for v in results.values()])


    mask = (BH > 0) & (Z > 0) & (Mstar > 0) & np.isfinite(BH) & np.isfinite(Z) & np.isfinite(Mstar)


    # BH vs Metallicity
    plt.figure(figsize=(8,6))
    plt.hexbin(Z[mask], BH[mask], gridsize=30, cmap='viridis',
               xscale='log', yscale='log', bins='log')
    plt.colorbar(label='log10(counts)')
    plt.xlabel(r"Mass-Weighted Gas Metallicity [$Z/Z_\odot$]")
    plt.ylabel(r"BH Primary Mass [$M_\odot$]")
    plt.tight_layout()
    plt.savefig("BHMass_vs_Metallicity.png")
    logging.info("Saved BHMass_vs_Metallicity.png")

    # BH vs Stellar Mass — aperture vs Subfind comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].hexbin(Mstar[mask], BH[mask], gridsize=30, cmap='viridis',
                   xscale='log', yscale='log', bins='log')
    axes[0].set_xlabel(r"Stellar Mass [$M_\odot$]")
    axes[0].set_ylabel(r"BH Primary Mass [$M_\odot$]")
    plt.tight_layout()
    plt.savefig("BHMass_vs_StellarMass_comparison.png")
    logging.info("Saved BHMass_vs_StellarMass_comparison.png")

  

    return results

# ============================================================
if __name__ == "__main__":
    import argparse, shutil, os
    parser = argparse.ArgumentParser()
    parser.add_argument("snap_base", help="Path to simulation snapshot directory (also contains groups_NNN dirs)")
    parser.add_argument("feedback", choices=["FullFeedback", "WeakFeedback", "NoFeedback"], help="Feedback model")
    parser.add_argument("region", choices=["Rarepeak", "Normal1", "Normal2"], help="Simulation region")
    args = parser.parse_args()
    base = "./"
    base = "%s%s_%s/" % (base, args.region, args.feedback)
    
    extract_galaxy_properties(args.snap_base, base)
    pkl_dir = "%s_%s" % (args.region, args.feedback)
    os.makedirs(pkl_dir, exist_ok=True)
    for f in glob.glob("*.pkl"):
        shutil.move(f, os.path.join(pkl_dir, os.path.basename(f)))
