import numpy as np
import matplotlib.pyplot as plt
import pickle
from extract_mergers import MergerExtractor

# Code Purpose
# From the sink_particle file this code uses the extract_mergers.py code (written by Daxal Mehta) to
# produce the mergers.pkl file which contains a list of all mergers occuring. Each BH merger contains
# the redshift of the merger, the mass of the primary, the mass of the secondary,
# the time of the mergerm and the ID of the merger remnant. 

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


def plot_mergers(base, outfile="mergers.pkl", m2_threshold=1000.0):

    outfile = base+outfile
    print("outfile = ", outfile)
    # ---------------------------------------------------------
    # Extract merger catalogue (Step 1)
    # ---------------------------------------------------------
    extractor = MergerExtractor(base)
    mergers_raw = extractor.extract_mergers()

    if len(mergers_raw) == 0:
        print("No mergers found!")
        return

    # Convert to arrays
    z  = np.array([m["Redshift"] for m in mergers_raw])
    M1 = np.array([m["M1"] for m in mergers_raw])
    M2 = np.array([m["M2"] for m in mergers_raw])
    t  = np.array([m["Time"] for m in mergers_raw])
    IDs = np.array([m["SinkID"] for m in mergers_raw])

    # ---------------------------------------------------------
    # Apply threshold on secondary mass (M2)
    # ---------------------------------------------------------
    mask = M2 > m2_threshold
    z, M1, M2, t, IDs = z[mask], M1[mask], M2[mask], t[mask], IDs[mask]

    # ---------------------------------------------------------
    # Sort descending in redshift
    # ---------------------------------------------------------
    sort = np.argsort(z)[::-1]
    z, M1, M2, t, IDs = z[sort], M1[sort], M2[sort], t[sort], IDs[sort]

    # ---------------------------------------------------------
    # Compute chirp mass
    # ---------------------------------------------------------
    chirp = ((M1 * M2)**(3/5)) / ((M1 + M2)**(1/5))

    # ---------------------------------------------------------
    # Build cleaned merger catalogue for saving
    # ---------------------------------------------------------
    cleaned = []
    for i in range(len(z)):
        cleaned.append({
            "SinkID": int(IDs[i]),
            "Redshift": float(z[i]),
            "Time": float(t[i]),
            "M1": float(M1[i]),
            "M2": float(M2[i]),
            "ChirpMass": float(chirp[i]),
        })

    # ---------------------------------------------------------
    # Save cleaned mergers to mergers.pkl
    # ---------------------------------------------------------
    with open(outfile, "wb") as f:
        pickle.dump(cleaned, f)

    print(f"Saved {len(cleaned)} mergers into {outfile}")

    # ---------------------------------------------------------
    # Diagnostic plots (3 panels)
    # ---------------------------------------------------------
    cumulative = np.arange(1, len(z) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(9, 16), sharex=True)

    # Panel 1: cumulative
    axes[0].plot(z, cumulative, color="black")
    axes[0].invert_xaxis()
    axes[0].set_ylabel("Cumulative Mergers")
    axes[0].set_title("BH Mergers")

    # Panel 2: M2
    axes[1].scatter(z, M2, s=25, alpha=0.7, color="tab:red")
    axes[1].plot(z, M2, color="tab:red", alpha=0.6)
    axes[1].invert_xaxis()
    axes[1].set_ylabel(r"Merger Mass Gain [$M_{\odot}$]")
    axes[1].set_yscale("log")

    # Panel 3: chirp mass
    axes[2].scatter(z, chirp, s=25, alpha=0.7, color="tab:blue")
    axes[2].plot(z, chirp, color="tab:blue", alpha=0.6)
    axes[2].invert_xaxis()
    axes[2].set_xlabel("Redshift")
    axes[2].set_ylabel(r"Chirp Mass [$M_{\odot}$]")
    axes[2].set_yscale("log")

    plt.tight_layout()
    plt.savefig("Mergers.png")
    print("Saved Mergers.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("snap_base", help="Path to simulation snapshot directory")
    parser.add_argument("feedback", choices=["FullFeedback", "WeakFeedback", "NoFeedback"], help="Feedback model")
    parser.add_argument("region", choices=["Rarepeak", "Normal1", "Normal2"], help="Region")
    args = parser.parse_args()
    base = "./"
    base = "%s%s_%s/" % (base, args.region, args.feedback)
    plot_mergers(base)
