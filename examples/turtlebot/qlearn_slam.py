import random

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def recommendActions(self, state, default_action):
        permitted = denied = []
        temp_state = state[:3]

        temp = [temp_state.index(x) for x in temp_state if x <= 2]
        p = sum(temp)
        l = len(temp)

        if l ==3: #all 3 directions are too close
            pass
        elif p == 3: #Direction RIGHT and LEFT are too close
            permitted.append(0)
        elif p < 3 and l == 2:
            pass
        elif p< 3 and l==1:
            permitted.append(3-p)

        else:
            temp = [x for x in temp_state if x > 2]
            temp.sort(reverse=True)
            for s in temp:
                permitted.append(state.index(s))

        permitted.append(3)

        if state[3] < 2:
            denied.append(1)
            if 1 in permitted:
                permitted.remove(1)
        if state[4] < 2:
            denied.append(2)
            if 2 in permitted:
                permitted.remove(2)
        '''
        if state[3] == 0:
            return 3
        if state[4] == 0:
            return 4
        if (state[3] == 0 or state[4] == 0) and l > 0:
            return 3 # only rotate is permitted
        '''
        if default_action is not None:
            if len(denied) == 0 or ((len(denied) > 0) and (default_action not in denied)):
                #print(".")
                if default_action in permitted:
                    return default_action
        #print("Updating Action----"+str(permitted[0])+"-------------State:"+str(state))
        return permitted[0]
        #return permitted, denied

    def chooseAction(self, state, return_q=False):
        Qstate = ''.join(map(str, state))
        q = [self.getQ(Qstate, a) for a in self.actions]
        maxQ = max(q)
        if random.random() < self.epsilon:
            minQ = min(q)
            mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        #action = self.actions[i]

        action = self.recommendActions(state, self.actions[i])

        if return_q:  # if they want it, give it!
            return action, q
        return action


    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)