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
        self.Q = {}
        self.QL_iterations = 100
        self.state = None
        self.total_rewards = 0
        self.original_q_value = 0
        self.s_old = None
        self.a_old = None
        self.reward_old = None
        # Q-Learning Parameters
        self.learning_rate = 0.35
        self.discount_factor = 0.8
        self.epsilon = 0.5 # exploration rate

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.total_rewards = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Auxiliary variables
        # print "inputs: {}".format(inputs)
        can_go_left = (inputs['light'] == 'green' and inputs['oncoming'] == None)
        can_go_right = (inputs['light'] == 'green') or (inputs['light'] == 'red' and inputs['oncoming'] == None)
        can_go_forward = (inputs['light'] == 'green')
        location = self.env.agent_states[self]['location']
        destination = self.planner.destination
        distance = self.env.compute_dist(location, destination)
        delta_x = destination[0] - location[0]
        delta_y = destination[1] - location[1]
        delta = (delta_x, delta_y)
        can_execute = False;
        if (self.next_waypoint == 'forward' and can_go_forward) or \
            (self.next_waypoint == 'left' and can_go_left) or \
            (self.next_waypoint == 'right' and can_go_right):
            can_execute = True

        # TODO: Update state
        s = (inputs, self.next_waypoint)

        # TODO: Select action according to your policy
        # find optimal action
        Q = self.Q
        optimal_action = random.choice(Environment.valid_actions)
        q_max = 0
        if Q.has_key( (s, optimal_action) ):
            q_max = Q[(s, optimal_action)]
        for a in Environment.valid_actions:
            if Q.has_key((s,a)) and Q[(s,a)] > q_max:
                optimal_action = a
                q_max = Q[(s,a)]

        # balance between exploration & exploitation
        action = optimal_action if random.random() > self.epsilon else random.choice(Environment.valid_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        # find optimal q value
        q_max = 0 # TODO: is 0 okay? or should it be lower
        for a in Environment.valid_actions:
            for (s_,a_) in Q.keys():
                if a == a_ and s == s_ and Q.has_key([s_,a_]) and Q[(s_,a_)] > q_max:
                    q_max = Q[(s_,a_)]

        # calculating q_old using (s_old, a_old) values which are from stored from previous iteration
        q_old = self.original_q_value if not Q.has_key((s_old,a_old)) else Q[(s_old,a_old)]

        # if still in learning mode, update Q-table
        if self.QL_iterations > 0:
            self.QL_iterations -= 1
            Q[(s, a)] = q_old + self.learning_rate * (reward_old + self.discount_factor * q_max - q_old)

        # setting s_old, a_old, and reward_old for next iteration
        self.s_old = s
        self.a_old = a
        self.reward_old = reward

        # keeping track of total rewards for logging purposes
        self.total_rewards += reward

        # print logs
        print "LearningAgent.update(): QL_iteration = {}, deadline = {}, inputs = {}, action = {}, reward = {}".format(QL_iteration, deadline, inputs, action, reward)  # [debug]
        print "next_waypoint: {}".format(self.next_waypoint)
        # print "Q: {}".format(Q)
        # for printing the state on screen
        self.state = "s:{}  total rewards:{}  deadline:{} ".format(s, self.total_rewards, deadline)


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
