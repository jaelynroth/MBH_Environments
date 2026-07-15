### MBH merger pipeline for Romulus25

This procedure gathers all the relevant data necessary for running the MBHEnvCatalogGenerator_Romulus25 file. Most data is already stored in the existing Romulus BH catalog, while the rest is sourced either from the intermediary Romulus25 database or the particle data. The existing catalog and any other outputs are stored as pkl files in the pklfiles subdirectory, from which the MBHEnvCatalogGenerator_Romulus25.py can read. 

The procedure for creating the existing BH catalog from the raw simulation outputs is not publicly available, but likely similar in concept to that described in https://github.com/mtremmel/changa_bh_tools.git -- contact Michael Tremmel for more details. 

Pipeline: 
1. 