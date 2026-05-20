"""
Standalone script to parse sink particle binary files and write:
  - sink_particle.pkl   : PopIII stars (Type 0) and MBHs (Type 3)
  - popii_particles.pkl : PopII stellar clusters (Type 2)

Usage:
    python CreateSinkPickleFile.py
"""

import os
import struct
import pickle
import numpy as np

HUBBLE_PARAMETER = 0.67

# ---------------------------------------------------------------------------
# Type and SNeType functions (provided by simulation codebase)
# ---------------------------------------------------------------------------

def Type(t):
    if t == 0: return "PopIII"
    if t == 2: return "PopII"
    if t == 3: return "MBH"

def SNeType(mass, hubble_parameter):
    mass *= 1e10 / hubble_parameter
    if mass < 11:         return "No SNe"
    if 11 < mass <= 40:   return "TypeII SNe"
    if 40 < mass <= 140:  return "DCBH"
    if 140 < mass <= 260: return "PISN"
    if mass > 260:        return "DCBH"


# ---------------------------------------------------------------------------
# Struct format builder (mirrors BinaryReader.__init__)
# ---------------------------------------------------------------------------

def build_struct_format(feedback=False, array_length=1):
    if feedback:
        main = (
            "3d 3d 3d d d d d q i i i i d d d d d d d d i "
            "200d "
            "200d "
        )
    else:
        main = (
            "3d 3d 3d d d d d q i i i i d d d d d d i "
            "200d "
            "200d "
        )

    extra = "d d" if array_length == 1 else "350d 350d"
    fmt = (main + extra).replace(" ", "")
    return fmt, struct.calcsize(fmt)


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def read_sink_info(base, feedback=False, array_length=1, hubble_parameter=0.6774):

    struct_format, struct_size = build_struct_format(feedback, array_length)

    sink_base       = os.path.join(base, "sink_particle_info/")
    pickle_file     = os.path.join(base, "sink_particle.pkl")
    pickle_file_popii = os.path.join(base, "popii_particles.pkl")

    files = sorted(os.listdir(sink_base))

    # ------------------------------------------------------------------
    # Decide what to process
    # ------------------------------------------------------------------
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            saved = pickle.load(f)

        saved_files = saved["meta"]["files_used"]
        new_files   = [f for f in files if f not in saved_files]

        if len(new_files) == 0:
            print("No new files. Loading existing pickles — nothing to do.")
            return

        print(f"New files detected: {len(new_files)}. Running incremental update.")

        sink_particles = saved["data"]

        with open(pickle_file_popii, "rb") as f:
            popii_particles = pickle.load(f)["data"]

        files_to_process = new_files
        last_time = saved["meta"].get("last_time", -np.inf)

    else:
        print("No pickle found. Running full build.")
        sink_particles  = {}
        popii_particles = {}
        files_to_process = files
        last_time = -np.inf

    # ------------------------------------------------------------------
    # Parse binary files
    # ------------------------------------------------------------------
    for fname in files_to_process:
        filepath = os.path.join(sink_base, fname)
        print(f"Opening file: {filepath}")

        with open(filepath, "rb") as f:
            while True:
                # --- read timestamp ---
                chunk = f.read(8)
                if not chunk or len(chunk) < 8:
                    break
                time = struct.unpack("d", chunk)[0]
                if time < 0 or time > 1:
                    break

                # --- read particle count ---
                count_chunk = f.read(4)
                if len(count_chunk) < 4:
                    break
                num_sinks = struct.unpack("i", count_chunk)[0]
                if num_sinks < 0:
                    break

                # --- skip already-processed timesteps ---
                if time <= last_time:
                    f.seek(struct_size * num_sinks, 1)
                    continue

                print(f"  Time: {time:.6f} | Sink particles: {num_sinks}")

                for _ in range(num_sinks):
                    try:
                        raw = f.read(struct_size)
                        if len(raw) < struct_size:
                            print(f"  Incomplete entry at time {time}, skipping remainder of timestep.")
                            break
                        data = struct.unpack(struct_format, raw)
                    except struct.error as e:
                        print(f"  struct.error at time {time}: {e}")
                        break

                    sink_id = data[13]

                    # Guard against corrupt IDs
                    if sink_id < 0:
                        print(f"  Warning: negative sink_id {sink_id} at time {time}, skipping entry.")
                        continue

                    entry = {
                        "Time":        time,
                        "Pos":         [data[0:3]],
                        "Vel":         [data[3:6]],
                        "Type":        data[17],
                        "StellarMass": data[23] if feedback else data[21],
                        "MergerMass":  data[24] if feedback else data[22],
                    }

                    # PopIII (0) and MBH (3) -> sink_particles
                    if entry["Type"] in (0, 3):
                        if sink_id not in sink_particles:
                            print(f"  Adding sink particle ID {sink_id} (Type {entry['Type']})")
                            sink_particles[sink_id] = {
                                "meta": {
                                    "StellarLifeTime": data[19],
                                    "FormationTime":   data[12],
                                    "Type":    Type(data[17]),
                                    "SNeType": SNeType(data[9], hubble_parameter) if data[17] == 0 else Type(data[17]),
                                    "Status":  None,
                                },
                                "evolution": {}
                            }
                        sink_particles[sink_id]["evolution"][time] = entry

                    # PopII (2) -> popii_particles
                    elif entry["Type"] == 2:
                        if sink_id not in popii_particles:
                            print(f"  Adding PopII particle ID {sink_id}")
                            popii_particles[sink_id] = {
                                "meta": {
                                    "StellarLifeTime": data[19],
                                    "FormationTime":   data[12],
                                    "Type":   Type(data[17]),
                                    "Status": None,
                                },
                                "evolution": {}
                            }
                        popii_particles[sink_id]["evolution"][time] = entry

                last_time = time

    # ------------------------------------------------------------------
    # Sort evolution dicts and assign final status
    # ------------------------------------------------------------------
    all_times = []
    for sid in sink_particles:
        sink_particles[sid]["evolution"] = dict(
            sorted(sink_particles[sid]["evolution"].items())
        )
        all_times.extend(sink_particles[sid]["evolution"].keys())

    for sid in popii_particles:
        popii_particles[sid]["evolution"] = dict(
            sorted(popii_particles[sid]["evolution"].items())
        )

    if not all_times:
        print("No sink particle data found — check your binary files.")
        return

    final_time = max(all_times)

    for sid, sp in sink_particles.items():
        meta      = sp["meta"]
        last_pres = final_time in sp["evolution"]
        if meta["Type"] == "PopIII":
            meta["Status"] = "In Simulation" if last_pres else (
                "Deleted" if meta["SNeType"] == "PISN" else "Merged"
            )
        elif meta["Type"] == "MBH":
            meta["Status"] = "In Simulation" if last_pres else "Merged"

    for sid, pp in popii_particles.items():
        last_pres = final_time in pp["evolution"]
        pp["meta"]["Status"] = "In Simulation" if last_pres else "Merged"

    # ------------------------------------------------------------------
    # Write pickles
    # ------------------------------------------------------------------
    metadata = {
        "files_used": files,
        "last_time":  final_time,
    }

    with open(pickle_file, "wb") as f:
        pickle.dump({"data": sink_particles, "meta": metadata}, f)
    print(f"Written: {pickle_file}")

    with open(pickle_file_popii, "wb") as f:
        pickle.dump({"data": popii_particles, "meta": metadata}, f)
    print(f"Written: {pickle_file_popii}")


# ---------------------------------------------------------------------------
# Entry point — edit these variables before running
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BASE             = "/home/daxal/data/ProductionRuns/Renaissance/NoFeedback/"
    FEEDBACK         = False
    ARRAY_LENGTH     = 1                       # 1 or 350
    HUBBLE_PARAMETER = 0.6774                  # e.g. Planck 2015 value

    read_sink_info(base=BASE, feedback=FEEDBACK, array_length=ARRAY_LENGTH, hubble_parameter=HUBBLE_PARAMETER)
