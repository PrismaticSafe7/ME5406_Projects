import FrozenLakeEnv as fl
import Trainer as tr
import pandas as pd
import numpy as np

class FrozenLake:
    def __init__(self, mapType="Default", mapSize=4):
        self.env = fl.FrozenLakeEnv(mapType,mapSize)
        self.data_collected = {}
    
    def runMC(self, no_of_runs=1, Test=False):
        MonteCarlo = tr.FVMonteCarlo(self.env)
        data_df = pd.DataFrame()
        datas = []

        for i in range(no_of_runs):
            new_df, data = MonteCarlo.train(i+1)
            data_df.append(new_df)
            datas.append(data)
            if i != no_of_runs - 1:
                MonteCarlo.Q_table = np.zeros((self.num_state, self.num_action))
                MonteCarlo.G_table = {(s,a): [] for s in range(self.num_state) for a in range(self.num_action)}
                MonteCarlo.policy = np.zeros(self.num_state, dtype=int)
        
        if Test:
            MonteCarlo.test()
        
        self.dataManager("MC.csv", "FVMC", data_df, no_of_runs)


    def runSARSA(self, no_of_runs=1, Test=False):
        SARSA = tr.SARSA(self.env)
        
        data_df = pd.DataFrame()
        datas = []

        for i in range(no_of_runs):
            new_df, data = SARSA.train(i+1)
            data_df.append(new_df)
            datas.append(data)
            if i != no_of_runs - 1:
                SARSA.Q_table = np.zeros((self.num_state, self.num_action))
                SARSA.policy = np.zeros(self.num_state, dtype=int)
        
        if Test:
            SARSA.test()
        
        self.dataManager("SARSA.csv", "SARSA", data_df, no_of_runs)

    def runQL(self, no_of_runs=1, Test=False):
        QL = tr.QLearning(self.env)

        data_df = pd.DataFrame()
        datas = []

        for i in range(no_of_runs):
            new_df, data = QL.train(i+1)
            data_df.append(new_df)
            datas.append(data)
            if i != no_of_runs - 1:
                QL.Q_table = np.zeros((self.num_state, self.num_action))
                QL.policy = np.zeros(self.num_state, dtype=int)
        
        if Test:
            QL.test()
        
        self.dataManager("QL.csv", "QL", data_df, no_of_runs)

        pass
    
    def dataManager(self, output_file, name, data_df, no_of_runs):
        new_acc_lst = []
        new_step_lst = []
        new_rew_lst = []

        for i in range(no_of_runs):
            accuracy = name + "_Accuracy" + str(no_of_runs + 1)
            steps = name + "_noSteps" + str(no_of_runs + 1)
            reward = name + "_Rewards" + str(no_of_runs + 1)

            new_acc = "Moving_Average_Accuracy" + str(no_of_runs + 1)
            new_step = "Moving_Average_Step" + str(no_of_runs + 1)
            new_rew = "Moving_Average_Reward" + str(no_of_runs + 1)

            new_acc_lst.append(new_acc)
            new_step_lst.append(new_step)
            new_rew_lst.append(new_rew)
            
            data_df[new_acc] = data_df[accuracy].rolling(100).mean()
            data_df[new_step] = data_df[steps].rolling(100).mean()
            data_df[new_rew] = data_df[reward].rolling(100).mean()
        
        data_df["Overall_Accuracy"] = data_df[new_acc_lst].mean(axis=1)
        data_df["Overall_Steps"] = data_df[new_step_lst].mean(axis=1)
        data_df["Overall_Reward"]  = data_df[new_rew_lst].mean(axis=1)

        data_df.to_csv(output_file)


if __name__ == "__main__":
    # Initalize Frozen_Lake Class
    frozenLake = FrozenLake()
    frozenLake.runMC(5)
    frozenLake.runSARSA(5)
    frozenLake.runQL(5)