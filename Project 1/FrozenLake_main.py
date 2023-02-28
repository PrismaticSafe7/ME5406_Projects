import FrozenLakeEnv as fl
import Trainer as tr

env = fl.FrozenLakeEnv("Default", 4) # 4x4 default environment for Frozen Lake
# env_new_10x = fl.FrozenLakeEnv("newMap", 10) # 10x10 generated map for Frozen Lake
# env_default_10x = fl.FrozenLakeEnv("Default", 10) # 10x10 default environment for Frozen Lake

class FrozenLake:
    def __init__(self, mapType="Default", mapSize=4):
        self.env = fl.FrozenLakeEnv(mapType,mapSize)
        self.data_collected = {}
    
    def runMC(self):
        MonteCarlo = tr.FVMonteCarlo(self.env)

        if 
        pass

    def runSARSA(self):
        pass

    def runQL(self):
        pass
        