import numpy as np
from DataReader import Reader
from Utilities import DataUtilities, Constants

CS = Constants()
DU = DataUtilities()

class MergerExtractor:
    def __init__(self, base):
        self.base = base
        self.reader = Reader(base)
        self.sinks = self.reader.pickle_reader("sink_particle.pkl")["data"]

    def extract_mergers(self):
        catalog = []

        for ID, sink in self.sinks.items():
            evo = sink["evolution"]
            times = np.array(sorted(evo.keys()))

            MergerMass = np.array([evo[t]["MergerMass"] for t in times])
            StellarMass = np.array([evo[t]["StellarMass"] for t in times])

            # Identify positive jumps = mergers
            deltas = np.diff(MergerMass)
            idx = np.where(deltas > 0)[0]


            #StellarMass in now the remnant mass
            #MergerMass will be the secondary mass (guaranteed by merger algorithm in the code)
            for j in idx:
                RemnantMass = StellarMass[j]
                SecondaryMass = deltas[j]
                PrimaryMass = RemnantMass - SecondaryMass
                entry = {
                    "SinkID": ID,
                    "Time": times[j],
                    "Redshift": 1/times[j] - 1,
                    "M1": PrimaryMass * 1e10 / CS.hubble_parameter,
                    "M2": SecondaryMass * 1e10 / CS.hubble_parameter,
                    "Mr": RemnantMass * 1e10 / CS.hubble_parameter,
                }
                catalog.append(entry)

        return catalog
