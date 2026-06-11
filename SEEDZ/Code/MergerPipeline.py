#!/usr/bin/env python3
"""
Pipeline order: 
python CreateSinkPickleFile.py <path to snapshots>
python DetermineMergers.py
python BHsGalaxies.py <path_to_snapshots>
python extract_galaxy-properties_parallel.py  <path_to_snapshots>
python MBHEnvCatalogGenerator_SEEDZ.py
(Optional) python PlotCatalogueContents.py
BHs2Galaxies -> extract_galaxy_properties -> MBHEnvCatalogGenerator
"""

import subprocess
import sys

# ============================================================
# CONFIGURATION
# ============================================================

#path_to_snapshots = "/home/daxal/data/ProductionRuns/Renaissance/Normal1/0.5Mpc/NoFeedback/"

#Vanilla Feedback
Feedback = "FullFeedback"

#Region = "Rarepeak"
#path_to_snapshots = "/home/daxal/data/ProductionRuns/Renaissance/Feedback_Elliptical/"
#Region = "Normal1"
#path_to_snapshots = "/home/daxal/data/ProductionRuns/Renaissance/Normal1/0.5Mpc/Feedback/"
#Region = "Normal2"
#path_to_snapshots = "/home/daxal/data/ProductionRuns/Renaissance/Normal2/0.5Mpc/Feedback/"

#Weak Feedback
Feedback = "WeakFeedback"

#Region = "Rarepeak"
#path_to_snapshots = "/home/daxal/data/ProductionRuns/Renaissance/WeakFeedback_Elliptical/"

Region = "Normal1"
path_to_snapshots = "/home/daxal/data/ProductionRuns/Renaissance/Normal1/0.5Mpc/WeakFeedback/"

SCRIPTS = [
    ["python", "CreateSinkPickleFile.py", path_to_snapshots, Feedback],
    ["python", "DetermineMergers.py"],
    ["python", "BHs2Galaxies.py",                              path_to_snapshots],
    ["python", "extract_galaxy_properties_parallel.py",  path_to_snapshots, Feedback, Region],
    ["python", "MBHEnvCatalogGenerator_SEEDZ.py", Feedback],
    ["python", "../../PlotCatalogueContents.py"]
]

# ============================================================

def run_step(cmd, step_num, total):
    script_name = cmd[1]
    print(f"\n{'='*60}")
    print(f"[{step_num}/{total}] Running: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n[FAILED] {script_name} exited with return code {result.returncode}. Stopping pipeline.")
        sys.exit(result.returncode)

    print(f"\n[OK] {script_name} completed successfully.")


if __name__ == "__main__":
    total = len(SCRIPTS)
    for i, cmd in enumerate(SCRIPTS, start=1):
        run_step(cmd, i, total)

    print(f"\n{'='*60}")
    print("Pipeline completed successfully.")
    print(f"{'='*60}")
