import random
import copy
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        Q_old = {}

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.total_reward = 0
        Q_old = {}

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Auxiliary variables
        # print "inputs: {}".format(inputs)
        can_go_left = (inputs['light'] == 'green' and inputs['oncoming'] == None)
        can_go_right = (inputs['light'] == 'green') or (inputs['light'] = 'red' and inputs['oncoming'] == None)
        can_go_forward = (inputs['light'] == 'green')
        location = self.env.agent_states[self]['location']
        destination = self.planner.destination
        distance = self.env.compute_dist(location, destination)

        # TODO: Update state
        # s = (distance, deadline, next_waypoint, can_go_left, can_go_forward, can_go_right)
        s = (next_waypoint, inputs, deadline - distance)
        
        # TODO: Select action according to your policy
        # find optimal action
        optimal_action = random.choice(Environment.valid_actions)
        q_max = Q_old[(s, optimal_action)]
        for a in Environment.valid_actions:
            if Q_new(s,a) > q:
                optimal_action = a
                q_max = Q_new(s,a)

        # Execute action and get reward
        action = optimal_action
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        # Initial population of Q_old
        first_run = True #set to detect first iteration of Q-learning

        # Q-Learning Parameters
        self.learning_rate = 0.2
        self.discount_factor = 0.8

        # Next states
        next_states = []
        optimal_future_value = # TODO: find max
        new_state = # TODO: define new state
        optimal_future_value = 0
        for k in Q_old.keys():
            s_ , a_ = k
            if(s_ != new_state):
                pass
            if Q_old[(s_,a_)] > optimal_future_value
                optimal_future_value = Q_old[(s_,a_)]

        q_old = 0
        if( Q_old.has_key((s,a))):
            q_old = Q_old[(s,a)]

        Q_new = copy.deepcopy(Q_old)
        Q_new[(s, a)] = q_old + self.learning_rate * (reward + self.discount_factor * optimal_future_value - q_old)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
