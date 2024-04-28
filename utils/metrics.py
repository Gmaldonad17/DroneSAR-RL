from math import sqrt
import matplotlib.pyplot as plt

class Metrics():
    def __init__(self):
        self.rewardsPerEpisode = []
        self.avgDistanceFromTarget = []
        self.avgStepsToComplete = []
        self.avgStepsToFindHint = []
        self.hintsFound = 0

        
    def FirstClueFind(self, environment):
        # Get number of clues found so far
        totalFoundClues = FoundClues(environment)
        # Check if no clues have been found before and at least one was just found
        if self.hintsFound == 0 and totalFoundClues > 0:
            # Save number of time steps it took to find it
            self.avgStepsToFindHint.append(environment.time_steps)
            # Save internally how many clues have been found
            self.hintsFound = totalFoundClues


    # Create graphs showing all results
    def GraphResults(self):
        # Plot Reward per episode
        plt.figure(1)
        plt.plot(self.rewardsPerEpisode, label = 'Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward VS Episode')
        plt.show()

        # Plot Average Distance per episode
        plt.figure(2)
        plt.plot(self.avgDistanceFromTarget, label = 'Reward')
        plt.xlabel('Episode')
        plt.ylabel('Distance')
        plt.title('Average Distance From Objective VS Episode')
        plt.show()

        # Plot Average steps to complete per episode
        plt.figure(3)
        plt.plot(self.avgStepsToComplete, label = 'Reward')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Average Steps To Find Objective VS Episode')
        plt.show()

        # Plot Average steps to find first hint per episode
        plt.figure(4)
        plt.plot(self.avgStepsToFindHint, label = 'Steps to find hint')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Average Steps To Find Hint VS Episode')
        plt.show()

        
    def UpdateMetrics(self, environment, reward):
        self.rewardsPerEpisode.append(reward)
        # Get position of the objective
        xObj, yObj = environment.objective[0], environment.objective[1]
        total_distance = 0
        total_drones = 0
        # Go through each drone to find distance to objective
        for _, agent in enumerate(environment.drones):
            xDrone, yDrone = agent.position
            total_drones += 1
            total_distance += sqrt((xObj - xDrone)**2 + (yObj - yDrone)**2)
        # Average drone distance and add average distance to metrics
        self.avgDistanceFromTarget.append(total_distance/total_drones) 
        # Add number of steps it took to metrics
        # If Objective was found, save amount of time it took
        if environment.discovery_map[yObj,xObj] == 1:
            self.avgStepsToComplete.append(environment.time_steps)
        # Objective was not found, apply max time steps
        else:
            self.avgStepsToComplete.append(environment.terminal_time_steps)
        # Add reward to metrics

        # avg time to find hint
        # If it didn't find a hint, save max time steps
        if self.hintsFound == 0:
                self.avgStepsToFindHint.append(environment.terminal_time_steps) # Max out value
        # Reset hint count for next iteration
        self.hintsFound = 0

# Goes through each clue and determines if it has been found
# Returns total number of found clues
def FoundClues(environment):
    # Get all clue locations
    clueLocations = environment.clues
    cluesFound = 0
    # Go through each clue and check if it is found on discovery map
    for clue in clueLocations:
        x, y = clue[1], clue[0]
        # 1 if it is found
        if environment.discovery_map[y,x] == 1:
            cluesFound += 1
    # Return total number of clues found
    return cluesFound