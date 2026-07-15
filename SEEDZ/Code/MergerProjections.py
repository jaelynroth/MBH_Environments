#!/usr/bin/env python3
"""
Create YT ProjectionPlots of gas number density around BH mergers
One projection per merger, centered on merger position, with red X marking the merger
"""

import numpy as np
import pickle
import yt
import glob
import os
from yt.visualization.api import ProjectionPlot

# ============================================================
# UTILITIES
# ============================================================

def load_mergers(pkl_file):
    with open(pkl_file, "rb") as f:
       data = pickle.load(f)
    
    return data
def find_snapshot_file(snap_base, snap_number):
    """
    Find the first snapshot file for a given snapshot number
    Snapshots are typically stored as snap_NNN.0.hdf5, snap_NNN.1.hdf5, etc.
    We load snap_NNN.0.hdf5
    """
    pattern = f"{snap_base}/snapdir_{snap_number:03d}/snap_{snap_number:03d}.0.hdf5"
    if os.path.exists(pattern):
        return pattern
    else:
        raise FileNotFoundError(f"Could not find snapshot {snap_number} at {pattern}")

def load_snapshot(snap_base, snap_number):
    """Load snapshot with yt"""
    snap_file = find_snapshot_file(snap_base, snap_number)
    print(f"  Loading {snap_file}...")
    ds = yt.load(snap_file)
    return ds

# ============================================================
# PROJECTION PLOT
# ============================================================

def create_merger_projection(ds, merger, output_dir="merger_projections", width_physical_kpc=2.0):
    """
    Create a projection plot centered on merger position
    
    Parameters:
    -----------
    ds : yt dataset
    merger : dict from merger_galaxies.pkl
    output_dir : directory to save plots
    width_comoving_mpc : width of projection in comoving Mpc
    """
    
    gal_id = merger["GalaxyID"]
    snap = merger["Snapshot"]
    redshift = merger["Redshift"]
    pos_code = np.array(merger["Center_code"])
    Mstar = merger["StellarMass"]
    BHMass = merger["BHRemnantMass"]
    
    # Convert to yt center coordinate
    # Position is in code units, need to wrap in yt quantity
    center = ds.arr(pos_code, "code_length")
    
    # Width in comoving Mpc
    # yt handles unit conversions automatically
    width = (width_physical_kpc, "kpc")  # yt will interpret as comoving at this redshift

    sphere = ds.sphere(pos_code, (2.0*width_physical_kpc, "kpc"))

    #grab star particles from the sphere too and overplot them too to show the galaxy.
    Type4Mass = sphere[("PartType4", "Masses")].v.sum()

    masses = sphere[("PartType5", "Masses")].v
    bh_threshold = 1e3/1e10  # code masses
    star_mask = masses < bh_threshold
    bh_mask   = masses >= bh_threshold
    star_mass = masses[star_mask].sum()
    bh_mass   = masses[bh_mask].sum()
    Type5Mass = sphere[("PartType5", "Masses")].v.sum()
    TotalStellarMass = Type4Mass*1e10+star_mass*1e10
    print("Total Stellar Mass = %e Msun" % (Type4Mass*1e10+star_mass*1e10))
    print("Total BH mass = %e Msun" % (bh_mass*1e10))
    if(TotalStellarMass < 1.0):
        return None
    # Create projection along z-axis (x-y plane)
    print(f"  Creating projection for merger {gal_id} (snap {snap}, z={redshift:.2f})...")
    proj = ProjectionPlot(
        ds, 
        "z",  # project along z-axis
        ("gas", "number_density"),
        center=center,
        width=width,
        weight_field=("gas", "density"),
        data_source=sphere
    )
    
    # Style the projection
    proj.set_log(("gas", "number_density"), True)
    proj.set_cmap(("gas", "number_density"), "viridis")
    proj.annotate_title(f"Merger {gal_id}: Snapshot {snap}, z={redshift:.3f}")
   
    #proj.annotate_text((0.0,0.0), "X", coord_system="plot", text_args={"color": "red"})
    proj.annotate_sphere(pos_code, radius=(5, "pc"), 
                         circle_args={'color':'black', 'fill':True, 'linestyle':'solid','linewidth':2.5})
    proj.annotate_text((0.1, 0.9), "BH Mass = %e" % (BHMass), coord_system="axis")
    proj.annotate_text((0.1, 0.85), "Stellar Mass = %e" % (Mstar), coord_system="axis")
    proj.annotate_particles(width=width, ptype="PartType4", col='white', p_size=5, data_source=sphere)
    proj.annotate_particles(width=width, ptype="PartType5", col='white', p_size=5, data_source=sphere)
    # Save
    os.makedirs(output_dir, exist_ok=True)
    outfile = f"{output_dir}/merger_{gal_id:06d}.png"
    proj.save(outfile)
    print(f"  Saved {outfile}")
    
    return proj

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create YT projections around BH mergers")
    parser.add_argument("snap_base", help="Path to simulation snapshot directory (contains snapdir_* dirs)")
    parser.add_argument("pkl_file", help="Path to galaxy_properties.pkl")
    parser.add_argument("--output", default="merger_projections", 
                       help="Output directory (default: merger_projections)")
    parser.add_argument("--width", type=float, default=1.0,
                       help="Projection width in comoving kpc (default: 1.0)")
    parser.add_argument("--limit", type=int, default=1,
                       help="Only process first N mergers (useful for testing)")
    args = parser.parse_args()
    
    print(f"Loading mergers from {args.pkl_file}...")
    data = load_mergers(args.pkl_file)
    galaxyresults = {}
    max_id = max(galaxyresults.keys()) if galaxyresults else 0
    print("max_id = ", max_id)
    for k, v in data.items():
        galaxyresults[max_id + k + 1] = v
    print(f"Loaded {args.pkl_file}: {len(data)} galaxies")
    galaxy_ids = sorted(galaxyresults.keys())
    N = len(galaxy_ids)
    print("N = ", N)
    print("galaxyresults.keys() = ", galaxyresults.keys())
    print("galaxyresults[1] = ", galaxyresults[1])
    
    # Track unique snapshots to avoid reloading
    ds_cache = {}
    
    for i, mergernumber in enumerate(galaxy_ids):
        print("i = ", i)
        print("merger = ", mergernumber)
        print("galaxyresults[1] = ", galaxyresults[mergernumber])
        print("galaxyresults[mergernumber[Snashot]] = ", galaxyresults[mergernumber]["Snapshot"])
        merger =  galaxyresults[mergernumber]
        snap = galaxyresults[mergernumber]["Snapshot"]
        print(f"\n[{i+1}/{N}] Merger {merger['GalaxyID']} at snap {snap}")
        
        # Load snapshot only once
        if snap not in ds_cache:
            ds_cache[snap] = load_snapshot(args.snap_base, snap)
            ds = ds_cache[snap]
            
            # Create and save projection
            create_merger_projection(ds, merger, args.output, args.width)
            
        
    print(f"\n✓ Done! Projections saved to {args.output}/")
