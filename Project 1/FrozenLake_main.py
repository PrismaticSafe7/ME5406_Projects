import FrozenLakeEnv as fl
import Trainer as tr
import pandas as pd
import numpy as np

class FrozenLake:
    def __init__(self, mapType="Default", mapSize=4):
        self.env = fl.FrozenLakeEnv(mapType,mapSize)
        self.data_collected = {}
        print(self.env.probabilityMatrix)
    
    def runMC(self, no_of_runs=1, Test=False):
        MonteCarlo = tr.FVMonteCarlo(self.env)
        data_df = pd.DataFrame()
        datas = []

        for i in range(no_of_runs):
            new_df, data = MonteCarlo.train(i+1)
            data_df = pd.concat([data_df,new_df], axis=1)
            datas.append(data)
            print(MonteCarlo.policy)
            if i != no_of_runs - 1:
                MonteCarlo.Q_table = np.zeros((MonteCarlo.num_state, MonteCarlo.num_action))
                MonteCarlo.G_table = {(s,a): [] for s in range(MonteCarlo.num_state) for a in range(MonteCarlo.num_action)}
                MonteCarlo.policy = np.zeros(MonteCarlo.num_state, dtype=int)
        
        if Test:
            MonteCarlo.test()
        
        self.dataManager("MC_10.csv", "FVMC", data_df, no_of_runs)


    def runSARSA(self, no_of_runs=1, Test=False):
        SARSA = tr.SARSA(self.env)
        
        data_df = pd.DataFrame()
        datas = []

        for i in range(no_of_runs):
            new_df, data = SARSA.train(i+1)
            data_df = pd.concat([data_df,new_df], axis=1)
            datas.append(data)
            print(SARSA.policy)
            if i != no_of_runs - 1:
                SARSA.Q_table = np.zeros((SARSA.num_state, SARSA.num_action))
                SARSA.policy = np.zeros(SARSA.num_state, dtype=int)
        
        if Test:
            SARSA.test()
        
        self.dataManager("SARSA_10.csv", "SARSA", data_df, no_of_runs)

    def runQL(self, no_of_runs=1, Test=False):
        QL = tr.QLearning(self.env)

        data_df = pd.DataFrame()
        datas = []

        for i in range(no_of_runs):
            new_df, data = QL.train(i+1)
            data_df = pd.concat([data_df,new_df], axis=1)
            print(QL.policy)
            datas.append(data)
            if i != no_of_runs - 1:
                QL.Q_table = np.zeros((QL.num_state, QL.num_action))
                QL.policy = np.zeros(QL.num_state, dtype=int)
        
        if Test:
            QL.test()
        
        self.dataManager("QL_10.csv", "Q", data_df, no_of_runs)

        pass
    
    def dataManager(self, output_file, name, data_df, no_of_runs):
        new_acc_lst = []
        new_step_lst = []
        new_rew_lst = []

        for i in range(no_of_runs):
            accuracy = name + "_Accuracy" + str(i + 1)
            steps = name + "_noSteps" + str(i + 1)
            reward = name + "_Rewards" + str(i + 1)

            new_acc = "Moving_Average_Accuracy" + str(i + 1)
            new_step = "Moving_Average_Step" + str(i + 1)
            new_rew = "Moving_Average_Reward" + str(i + 1)

            new_acc_lst.append(new_acc)
            new_step_lst.append(new_step)
            new_rew_lst.append(new_rew)
            
            data_df[new_acc] = data_df[accuracy].rolling(100, min_periods = 0).mean()
            data_df[new_step] = data_df[steps].rolling(100,min_periods = 0).mean()
            data_df[new_rew] = data_df[reward].rolling(100,min_periods = 0).mean()
        
        data_df["Overall_Accuracy"] = data_df[new_acc_lst].mean(axis=1)
        data_df["Overall_Steps"] = data_df[new_step_lst].mean(axis=1)
        data_df["Overall_Reward"]  = data_df[new_rew_lst].mean(axis=1)

        data_df.to_csv(output_file)


if __name__ == "__main__":
    np.random.seed(12)

    # Initalize Frozen_Lake Class
    frozenLake = FrozenLake()
    frozenLake.runMC(1)
    frozenLake.runSARSA(1)
    frozenLake.runQL(1)

    # Frozen_Lake for 10x10
    # frozenLake = FrozenLake("Default",10)
    # frozenLake.runMC(1)
    # frozenLake.runSARSA(1)
    # frozenLake.runQL(1)