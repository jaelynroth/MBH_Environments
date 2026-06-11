import numpy as np
import pickle
import yt
import glob
from DataReader import Reader

# Code Purpose
# Connect the BH mergers (as determined from the sink pickle files to snapshots.
# Dumps out the merger_galaxies.pkl file which contains the merger snapshot
# (closest one to merger at least) number, the redshift of the snapshot,
# the BH position at that redshift, the BH ID (to connect to the mergers.pkl file)
# and a galaxy ID (dummy for now).
# Note that the same galaxy will be added to to the merger_galaxies file if a BH
# merger occurs in the same galaxy at a different redshift. 

# ---------------------------------------------------------
# Load redshifts of hydrodynamic snapshots
# ---------------------------------------------------------
def load_snapshot_redshifts(snap_base):
    snapdirs = sorted(glob.glob(f"{snap_base}/snapdir_*"))
    snap_info = []

    for path in snapdirs:
        snap = int(path.split("_")[-1])
        try:
            ds = yt.load(f"{path}/snap_{snap:03d}.0.hdf5")
            z = ds.current_redshift
            snap_info.append((snap, z))
        except Exception as e:
            print(f"[WARN] Could not load snapshot {snap}: {e}")
            continue

    snap_info.sort(key=lambda x: x[1], reverse=True)
    return snap_info


# ---------------------------------------------------------
# Find nearest snapshot that is before the merger redshift
# ---------------------------------------------------------
def nearest_snapshot(z, snap_list):
    before = [(snap, z_snap) for snap, z_snap in snap_list if z_snap > z]
    if not before:
        return min(snap_list, key=lambda x: abs(x[1] - z))[0]  # fallback: merger before first snapshot
    return min(before, key=lambda x: x[1])[0]  # nearest = lowest redshift in the before set


# ---------------------------------------------------------
# Get remnant BH position at nearest snapshot time
# ---------------------------------------------------------
def get_position_at_snapshot(ID, a_snap, sinks):
    """Find sink position at evolution time closest to snapshot scale factor."""
    sink = sinks[ID]
    times = np.array(sorted(sink["evolution"].keys()))
    t_sel = times[np.argmin(abs(times - a_snap))]
    pos = sink["evolution"][t_sel]["Pos"]
    return np.array(pos).reshape(-1)


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def build_merger_galaxies(
        pkl_base, snap_base, 
        mergers_file="mergers.pkl",
        outfile="merger_galaxies.pkl"
    ):

    # Load sinks
    R = Reader(pkl_base)
    sinks = R.pickle_reader("sink_particle.pkl")["data"]

    # Load mergers
    with open(mergers_file, "rb") as f:
        mergers = pickle.load(f)

    # Load snapshot redshifts
    snap_list = load_snapshot_redshifts(snap_base)
    if len(snap_list) == 0:
        raise RuntimeError("No hydrodynamic snapshots found!")

    galaxies = []
    galID = 0

    for m in mergers:
        primary_id = m["SinkID"]
        z_merge = m["Redshift"]

        # ---------------------------------------------------------
        # Match to nearest snapshot
        # ---------------------------------------------------------
        snap = nearest_snapshot(z_merge, snap_list)
        _, z_snap = next(x for x in snap_list if x[0] == snap)
        a_snap = 1/(1 + z_snap)

        # ---------------------------------------------------------
        # Get remnant BH position at that snapshot time
        # ---------------------------------------------------------
        try:
            pos = get_position_at_snapshot(primary_id, a_snap, sinks)
        except Exception as e:
            print(f"[WARN] Could not obtain position for BH {primary_id} at z={z_snap}: {e}")
            continue

        # ---------------------------------------------------------
        # Store galaxy entry
        # ---------------------------------------------------------
        galaxies.append({
            "GalaxyID": galID,
            "PrimaryID": primary_id,
            "Snapshot": snap,
            "Redshift": z_snap,
            "MergerTime": m["Time"],
            "Center": pos.tolist(),
        })

        galID += 1

    # Save results
    with open(outfile, "wb") as f:
        pickle.dump(galaxies, f)

    print(f"Saved {len(galaxies)} merger-hosting galaxies → {outfile}")
    return galaxies


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("snap_base", help="Path to simulation snapshot directory")
    args = parser.parse_args()
    pkl_base = "./"

    build_merger_galaxies(pkl_base, args.snap_base)
