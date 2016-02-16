import random
from collections import OrderedDict
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
        self.Q = OrderedDict()
        self.QL_iterations = 100
        self.state = None
        self.total_rewards = 0
        self.original_q_value = 0
        self.s_old = None
        self.a_old = None
        self.reward_old = 0
        # Q-Learning Parameters
        self.learning_rate = 0.4
        self.discount_factor = 0.7
        self.epsilon = self.QL_iterations / 100.0 # exploration rate
        self.distance = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # Run one more iteration of Q-Learning to capture the reward from last step
        q_max = 0
        s = self.s_old
        Q = self.Q
        for a in Environment.valid_actions:
            for (s_,a_) in Q.keys():
                if a == a_ and s == s_ and Q.has_key((s_,a_)) and Q[(s_,a_)] > q_max:
                    q_max = Q[(s_,a_)]
        q_old = self.original_q_value if not Q.has_key((self.s_old,self.a_old)) else Q[(self.s_old,self.a_old)]
        if self.QL_iterations > 0:
            Q[(self.s_old, self.a_old)] = q_old + self.learning_rate * (self.reward_old + self.discount_factor * q_max - q_old)


        # Change epsilon for each trial (exploration rate)
        self.epsilon = self.QL_iterations / 100.0 # exploration rate

        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.total_rewards = 0
        if self.QL_iterations > 0:
            self.QL_iterations -= 1

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        # Auxiliary variables
        location = self.env.agent_states[self]['location']
        destination = self.planner.destination
        self.distance = self.env.compute_dist(location, destination)
        # TODO: Update state
        s = (self.next_waypoint, inputs['light'], inputs['left'], inputs['oncoming'], self.distance)
        self.state = s
        # run QL_iterate step
        self.QL_iterate(s)


    def QL_iterate(self, state):
        s = state
        # TODO: Select action according to your policy
        # find optimal action
        Q = self.Q
        optimal_action = random.choice(Environment.valid_actions)
        q_max = 0
        if Q.has_key((s, optimal_action)):
            q_max = Q[(s, optimal_action)]
        for a in Environment.valid_actions:
            if Q.has_key((s,a)) and Q[(s,a)] > q_max:
                optimal_action = a
                q_max = Q[(s,a)]

        # Balance between exploration & exploitation
        action = optimal_action if random.random() > self.epsilon else random.choice(Environment.valid_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        # find optimal q value
        q_max = 0
        for a in Environment.valid_actions:
            for (s_,a_) in Q.keys():
                if a == a_ and s == s_ and Q.has_key((s_,a_)) and Q[(s_,a_)] > q_max:
                    q_max = Q[(s_,a_)]

        # Calculating q_old using (s_old, a_old) values which are from stored from previous iteration
        q_old = self.original_q_value if not Q.has_key((self.s_old,self.a_old)) else Q[(self.s_old,self.a_old)]

        # If still in learning mode, update Q-table
        if self.QL_iterations > 0:
            Q[(self.s_old, self.a_old)] = q_old + self.learning_rate * (self.reward_old + self.discount_factor * q_max - q_old)

        # setting s_old, a_old, and reward_old for next iteration
        self.s_old = s
        self.a_old = action
        self.reward_old = reward

        # keeping track of total rewards for logging purposes
        self.total_rewards += reward

        # uncomment for debug:
        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(self.env.get_deadline(self), self.env.sense(self), action, reward)  # [debug]


    def print_Q(self): #for debuggin
        print "Q = {"
        for key in self.Q.keys():
            print "{}: {}".format(key, self.Q[key])
        print "}\n"


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=120)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()