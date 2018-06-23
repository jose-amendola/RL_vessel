import experiment
import os

if __name__ == "__main__":
    agents = ['agents/agent_20180524154344Sequential_r_linear_with_rudder_punish_disc_0.8it5.h5']
    for agent in agents:
        experiment.evaluate_agent(agent)
        os.chdir('..')
