import pickle
import matplotlib.pyplot as plt
import numpy as np
with open("galaxy_properties.pkl", "rb") as f:
    results = pickle.load(f)


# ---- Plots ----
BH = np.array([v["BHRemnantMass"] for v in results.values()])
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
plt.ylabel(r"BH Remnant Mass [$M_\odot$]")
plt.tight_layout()
plt.savefig("BHMass_vs_Metallicity.png")



plt.figure(figsize=(8,6))
plt.scatter(MStellar, BH, s=20, alpha=0.7, color="tab:blue")
#plt.xlim(1e-5, 1e0)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Stellar Mass [$M_\odot$]")
plt.ylabel(r"BH Remnant Mass [$M_\odot$]")
plt.tight_layout()
plt.savefig("BHMass_vs_StellarMass.png")

