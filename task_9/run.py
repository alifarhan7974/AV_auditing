import sys 

# Wrapper functions 
from dqn_testing import run_dqn 
from a3c_testing import run_a3c 


def benchmarking(dqn_steps, dqn_episodes, a3c_steps):
    run_dqn(dqn_steps, dqn_episodes)
    run_a3c(max_global_steps=a3c_steps)

if __name__ == "__main__":

    # Returns with exit code 1 if incorrect input 
    if len(sys.argv) != 4: 
        print("USAGE: python3 run.py dqn_steps dqn_episodes a3c_steps")
        sys.exit(1) # exit code of 1 like in c 

    # Have to cast to an int
    dqn_steps = int(sys.argv[1])
    dqn_episodes = int(sys.argv[2])
    a3c_steps = int(sys.argv[3])

    benchmarking(dqn_steps, dqn_episodes, a3c_steps)




