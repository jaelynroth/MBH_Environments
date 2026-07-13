import h5py

import yt
filename = "/home/daxal/data/ProductionRuns/Renaissance/Normal1/0.5Mpc/NoFeedback/groups_042/fof_subhalo_tab_042.0.hdf5"
ds_subfind = yt.load(filename)
print(ds_subfind.field_list)



with h5py.File(filename, "r") as f:
    def print_structure(name, obj):
        print(name)
    f.visititems(print_structure)


import numpy as np

with h5py.File(filename, "r") as f:
    print("Header:", dict(f["Header"].attrs))
    print("SubhaloPos shape:", f["Subhalo/SubhaloPos"].shape)
    print("SubhaloPos[0:3]:", f["Subhalo/SubhaloPos"][:3])
    print("SubhaloMassType shape:", f["Subhalo/SubhaloMassType"].shape)
    print("SubhaloMassType[0:3]:", f["Subhalo/SubhaloMassType"][:3])
    print("SubhaloHalfmassRadType[0:3]:", f["Subhalo/SubhaloHalfmassRadType"][:3])
    print("Group_R_Crit200[0:3]:", f["Group/Group_R_Crit200"][:3])
    print("Group_M_Crit200[0:3]:", f["Group/Group_M_Crit200"][:3])
