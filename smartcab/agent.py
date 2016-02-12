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
        # TODO: can't have dict in state since it's unhashable
        # s = (self.next_waypoint, inputs, deadline)
        s = (self.next_waypoint, can_execute, deadline)
        
        # for printing the state on screen
        self.state = "s:{}  total rewards:{}  deadline:{} ".format(s, self.total_rewards, deadline) 


        # TODO: Select action according to your policy
        Q = self.Q
        optimal_action = random.choice(Environment.valid_actions)
        q_max = 0
        if Q.has_key( (s, optimal_action) ):
            q_max = Q[(s, optimal_action)]
        for a in Environment.valid_actions:
            if Q.has_key((s,a)) and Q[(s,a)] > q_max:
                optimal_action = a
                q_max = Q[(s,a)]

        # Execute action and get reward
        action = optimal_action
        reward = self.env.act(self, action)
        self.total_rewards += reward

        # TODO: Learn policy based on state, action, reward

        # Q-Learning Parameters
        self.learning_rate = 0.35
        self.discount_factor = 0.8

        # calculating optimal future value for s(t+1) used in Q learning , in this case an expected value
        # since future state is not deterministic and could have 2^8 = 256 outcomes

        total_q = {}
        for a in Environment.valid_actions:
            total_q[a] = 0
            for (s_,a_) in Q.keys():
                if s_[2] != deadline-1:
                    pass
                else:
                    print "s_: {}".format(s_)
                    total_q[a] += Q[(s_,a_)]
            total_q[a] /= 32 # getting the expected value

        optimal_future_value = max(total_q.values())

        print "\n\nnext_waypoint: {}".format(self.next_waypoint)
        # print "Q: {}".format(Q)
        print "self.QL_iterations: {}".format(self.QL_iterations)
        print "optimal_action: {}".format(optimal_action)
        print "optimal_future_value: {}".format(optimal_future_value)

        q_old = 0
        if( Q.has_key((s,a)) ):
            q_old = Q[(s,a)]

        if self.QL_iterations > 0:
            self.QL_iterations -= 1
            Q[(s, a)] = q_old + self.learning_rate * (reward + self.discount_factor * optimal_future_value - q_old)

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
