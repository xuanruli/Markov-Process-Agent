
# 1. Value Iteration
class ValueIterationAgent:
    """Implement Value Iteration Agent using Bellman Equations."""

    def __init__(self, game, discount):
        """Store game object and discount value into the agent object,
        initialize values if needed.
        """
        self._discount = discount
        self._game = game
        self._value = {state:0 for state in game.states}


    def get_value(self, state):
        """Return value V*(s) correspond to state.
        State values should be stored directly for quick retrieval.
        """
        return self._value.get(state, 0)


    def get_q_value(self, state, action):
        """Return Q*(s,a) correspond to state and action.
        Q-state values should be computed using Bellman equation:
        Q*(s,a) = Σ_s' T(s,a,s') [R(s,a,s') + γ V*(s')]
        """
        q_value = 0
        transition = self._game.get_transitions(state, action)
        for next_state, prob in transition.items():
            reward = self._game.get_reward(state, action, next_state)
            q_value += prob * (reward + self._discount * self.get_value(next_state))

        return q_value

    def get_best_policy(self, state):
        """Return policy π*(s) correspond to state.
        Policy should be extracted from Q-state values using policy extraction:
        π*(s) = argmax_a Q*(s,a)
        """
        actions = self._game.get_actions(state)
        if not actions:
            return None
        return max(actions, key = lambda action: self.get_q_value(state, action))

    def iterate(self):
        """Run single value iteration using Bellman equation:
        V_{k+1}(s) = max_a Q*(s,a)
        Then update values: V*(s) = V_{k+1}(s)
        """
        new_values = {}
        for state in self._game.states:
            actions = self._game.get_actions(state)
            if not actions: # termination
                new_values[state] = self._value[state]
            else:
                new_values[state] = max(self.get_q_value(state, action) for action in actions)

        self._value = new_values


# 2. Policy Iteration
class PolicyIterationAgent(ValueIterationAgent):
    """Implement Policy Iteration Agent.

    The only difference between policy iteration and value iteration is at
    their iteration method. However, if you need to implement helper function
    or override ValueIterationAgent's methods, you can add them as well.
    """
    def __init__(self, game, discount):
        super().__init__(game, discount)
        self._policy = {state: None for state in game.states}

    def evaluate_policy(self, epsilon=1e-6):
        while True:
            delta = 0
            new_values = self._value.copy()
            for state in self._game.states:
                action = self._policy[state]
                if action is not None:
                    new_value = self.get_q_value(state, action)
                    delta = max(delta, abs(new_values[state] - new_value))
                    new_values[state] = new_value
            self._value = new_values
            if delta < epsilon:
                break

    def improve_policy(self):
        stable = True
        for state in self._game.states:
            best_action = self.get_best_policy(state)
            if best_action != self._policy.get(state):
                self._policy[state] = best_action
                stable = False
        return stable

    def iterate(self):
        """Run single policy iteration.
        Fix current policy, iterate state values V(s) until
        |V_{k+1}(s) - V_k(s)| < ε
        """
        epsilon = 1e-6
        self.evaluate_policy(epsilon=epsilon)
        return self.improve_policy()


# 3. Bridge Crossing Analysis
def question_3():
    discount = 0.9
    noise = 0.00001
    return discount, noise


# 4. Policies
def question_4a():
    discount = 0.9
    noise = 0.2
    living_reward = -3
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4b():
    discount = 0.5
    noise = 0.4
    living_reward = -1
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4c():
    discount = 0.9
    noise = 0.05
    living_reward = -1
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4d():
    discount = 0.9
    noise = 0.4
    living_reward = -0.5
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4e():
    discount = 0.9
    noise = 0
    living_reward = 2
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'
