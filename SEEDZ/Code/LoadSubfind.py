import yt
filename = "/home/daxal/data/ProductionRuns/Renaissance/Normal1/0.5Mpc/NoFeedback/groups_042/fof_subhalo_tab_042.0.hdf5"
ds = yt.load(filename)

# Check all available particle types
print("Particle types:", ds.particle_types)

# Try accessing subhalo data directly
print("\nAll fields:")
for f in ds.field_list:
    print(f)

# Try explicitly
ad = ds.all_data()
try:
    print("\nSubhaloPos:", ad[("Subhalo", "SubhaloPos")])
except Exception as e:
    print("SubhaloPos failed:", e)

try:
    print("\nSubhaloMassType:", ad[("Subhalo", "SubhaloMassType_5")])
except Exception as e:
    print("SubhaloMassType_5 failed:", e)

# Check derived fields too
print("\nDerived fields (subhalo-related):")
for f in ds.derived_field_list:
    if "ubhalo" in str(f):
        print(f)
