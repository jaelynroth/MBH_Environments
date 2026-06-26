#!/usr/bin/env python3
"""
Create YT ProjectionPlots of gas number density around BH mergers
One projection per merger, centered on merger position.

Parallelization strategy:
- Mergers are grouped by snapshot.
- Each (snapshot, its mergers) group is handed to ONE worker process,
  which loads that snapshot exactly once and works through all of its
  mergers serially.
- Different groups run in parallel across worker processes, so
  parallelism scales with the number of *distinct snapshots*, not with
  mergers-per-snapshot. This avoids ever loading the same snapshot more
  than once, and each worker opens its own file independently (no
  shared dataset / fork-COW dependency to worry about).

Caveat: if mergers are very unevenly distributed across snapshots (e.g.
one snapshot has 50 mergers while most have 1-2), the worker assigned
that heavy snapshot becomes the long pole while other workers finish
early and sit idle. The script prints the mergers-per-snapshot
distribution up front so you can see if that applies to your data.
"""

import numpy as np
import pickle
import yt
import os
import multiprocessing as mp
from collections import defaultdict
from yt.visualization.api import ProjectionPlot

import matplotlib
matplotlib.use("Agg")  # no display needed, and safer across subprocesses

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
    print(f"  [snap {snap_number}] Loading {snap_file}...")
    ds = yt.load(snap_file)
    return ds

# ============================================================
# PROJECTION PLOT
# ============================================================

def create_merger_projection(ds, merger, output_dir="merger_projections", width_physical_kpc=3.0):
    """
    Create a projection plot centered on merger position

    Parameters:
    -----------
    ds : yt dataset
    merger : dict from galaxy_properties.pkl
    output_dir : directory to save plots
    width_physical_kpc : width of projection in kpc
    """

    gal_id = merger["GalaxyID"]
    snap = merger["Snapshot"]
    redshift = merger["Redshift"]
    pos_code = np.array(merger["Center_code"])
    Mstar = merger["StellarMass"]
    BHMass = merger["BHRemnantMass"]

    center = ds.arr(pos_code, "code_length")
    width = (width_physical_kpc, "kpc")

    sphere = ds.sphere(pos_code, (2.0 * width_physical_kpc, "kpc"))

    Type4Mass = sphere[("PartType4", "Masses")].v.sum()

    masses = sphere[("PartType5", "Masses")].v
    bh_threshold = 1e3 / 1e10  # code masses
    star_mask = masses < bh_threshold
    bh_mask = masses >= bh_threshold
    star_mass = masses[star_mask].sum()
    bh_mass = masses[bh_mask].sum()
    TotalStellarMass= Type4Mass*1e10 + star_mass*1e10
    if(TotalStellarMass < 1.0):
        print(f"  [merger {gal_id}] Skipping: stellar mass {TotalStellarMass:.3e} Msun looks bogus")
        return None
    print(f"  [merger {gal_id}] Total Stellar Mass = {Type4Mass*1e10 + star_mass*1e10:e} Msun")
    print(f"  [merger {gal_id}] Total BH mass = {bh_mass*1e10:e} Msun")
    print(f"  [merger {gal_id}] Creating projection (snap {snap}, z={redshift:.2f})...")

    proj = ProjectionPlot(
        ds,
        "z",  # project along z-axis
        ("gas", "number_density"),
        center=center,
        width=width,
        weight_field=("gas", "density"),
        data_source=sphere
    )

    proj.set_log(("gas", "number_density"), True)
    proj.set_cmap(("gas", "number_density"), "viridis")
    proj.set_zlim(("gas", "number_density"), 1e4, 1e-2) 
    proj.annotate_sphere(pos_code, radius=(5, "pc"),
                          circle_args={'color': 'black', 'fill': True, 'linestyle': 'solid', 'linewidth': 2.5})
    proj.annotate_text((0.1, 0.95), "Redshift = %1.1f" % (redshift), coord_system="axis")
    proj.annotate_text((0.1, 0.9), "BH Mass = %1.1e" % (BHMass), coord_system="axis")
    proj.annotate_text((0.1, 0.85), "Stellar Mass = %1.1e" % (Mstar), coord_system="axis")
    proj.annotate_particles(width=width, ptype="PartType4", col='white', p_size=5, data_source=sphere)
    proj.annotate_particles(width=width, ptype="PartType5", col='white', p_size=5, data_source=sphere)

    os.makedirs(output_dir, exist_ok=True)
    outfile = f"{output_dir}/merger_{gal_id:06d}.png"
    proj.save(outfile)
    print(f"  [merger {gal_id}] Saved {outfile}")

    return outfile

# ============================================================
# WORKER: one snapshot group, start to finish
# ============================================================

def _process_snapshot_group(task):
    """
    Runs entirely inside one worker process: loads its assigned snapshot
    once, then produces a projection for every merger mapped to it.
    Self-contained -- no shared/inherited dataset, so this is safe
    regardless of multiprocessing start method.
    """
    snap_base, snap, mergers, output_dir, width = task
    results = []
    try:
        ds = load_snapshot(snap_base, snap)
    except Exception as e:
        # whole snapshot failed to load -- mark every merger in it as failed
        return [(m["GalaxyID"], False, f"snapshot load failed: {e!r}") for m in mergers]

    for merger in mergers:
        gal_id = merger["GalaxyID"]
        try:
            outfile = create_merger_projection(ds, merger, output_dir, width)
            results.append((gal_id, True, outfile))
        except Exception as e:
            results.append((gal_id, False, repr(e)))

    del ds
    return results

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create YT projections around BH mergers (parallel across snapshots)")
    parser.add_argument("snap_base", help="Path to simulation snapshot directory (contains snapdir_* dirs)")
    parser.add_argument("pkl_file", help="Path to galaxy_properties.pkl")
    parser.add_argument("--output", default="merger_projections",
                         help="Output directory (default: merger_projections)")
    parser.add_argument("--width", type=float, default=1.0,
                         help="Projection width in kpc (default: 1.0)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process first N mergers total (useful for testing)")
    parser.add_argument("--nproc", type=int, default=None,
                         help="Max worker processes / concurrent snapshots (default: all available cores)")
    args = parser.parse_args()

    print(f"Loading mergers from {args.pkl_file}...")
    data = load_mergers(args.pkl_file)

    # Preserved from the original script: remaps dict keys to a 1-indexed,
    # offset id space (currently a no-op offset of 0, kept in case this is
    # ever called with a pre-populated galaxyresults from multiple files).
    galaxyresults = {}
    max_id = max(galaxyresults.keys()) if galaxyresults else 0
    for k, v in data.items():
        galaxyresults[max_id + k + 1] = v
    print(f"Loaded {args.pkl_file}: {len(data)} galaxies")

    galaxy_ids = sorted(galaxyresults.keys())
    if args.limit is not None:
        galaxy_ids = galaxy_ids[:args.limit]
    N = len(galaxy_ids)
    print(f"N = {N}")

    # Group mergers by snapshot -- each group becomes one worker's job
    by_snapshot = defaultdict(list)
    for gid in galaxy_ids:
        merger = galaxyresults[gid]
        by_snapshot[merger["Snapshot"]].append(merger)

    n_snaps = len(by_snapshot)
    group_sizes = sorted((len(v) for v in by_snapshot.values()), reverse=True)
    ncores = args.nproc or os.cpu_count()

    print(f"\n{N} merger(s) across {n_snaps} snapshot(s)")
    print(f"Mergers per snapshot -- min: {group_sizes[-1]}, max: {group_sizes[0]}, "
          f"median: {group_sizes[len(group_sizes)//2]}")
    if group_sizes[0] > 3 * (N / max(n_snaps, 1)):
        print("  Note: distribution looks skewed -- one or more snapshots carry a "
              "disproportionate share of mergers, so the busiest worker may dominate "
              "total runtime regardless of core count.")

    nworkers = min(ncores, n_snaps)
    print(f"Using {nworkers} worker process(es) (cores requested/available: {ncores})\n")

    tasks = [(args.snap_base, snap, mergers, args.output, args.width)
              for snap, mergers in sorted(by_snapshot.items())]

    done = 0
    failures = []
    with mp.Pool(processes=nworkers) as pool:
        for group_results in pool.imap_unordered(_process_snapshot_group, tasks):
            for gal_id, ok, info in group_results:
                done += 1
                status = "OK" if ok else "FAILED"
                print(f"[{done}/{N}] merger {gal_id}: {status} ({info})")
                if not ok:
                    failures.append((gal_id, info))

    print(f"\n✓ Done! Projections saved to {args.output}/")
    if failures:
        print(f"\n⚠ {len(failures)} merger(s) failed:")
        for gal_id, info in failures:
            print(f"  merger {gal_id}: {info}")
