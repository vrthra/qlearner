from functools import reduce, lru_cache

import subprocess
import string
import json
import shutil
import os
import os.path
import sys
import random
R = int(os.getenv('R') or '0')
#random.seed(R)

All_Characters = [i for i in list(string.ascii_letters + string.digits + string.punctuation) if i != "'"]
#All_Characters = [i for i in list(string.printable) if i != "'"]
class QState:
    Counter = 0
    def __init__(self, key):
        self.key = key
        self._policy = QPolicy()
        self._id = QState.Counter
        QState.Counter += 1

    def __str__(self):
        return "[state:%d key:%s %s]" % (self._id, self.key, self._policy)

    def __repr__(self):
        return str(self)

    def to_obj(self):
        return ('QState', (self.key, self._policy.to_obj()))

    @classmethod
    def from_obj(self, o):
        name, (key, policy) = o
        assert name == 'QState'
        s = QState(key)
        s._policy = QPolicy.from_obj(policy)
        return s

    @staticmethod
    def get_key(chars):
        my_chars = []
        for c in chars:
            if c in string.ascii_letters:
                if my_chars and my_chars[-1] == 'a':
                    continue
                else:
                    my_chars.append('a')
            elif c in string.digits:
                if my_chars and my_chars[-1] == '1':
                    continue
                else:
                    my_chars.append('1')
            elif c in string.whitespace:
                if my_chars and my_chars[-1] == ' ':
                    continue
                else:
                    my_chars.append(' ')
            else:
                my_chars.append(c)
        return ''.join(my_chars)



class Q:
    def __init__(self):
        self.chars, self._q = All_Characters, {}

    def __getitem__(self, key):
        if key not in self._q: self._q[key] = 0
        return self._q[key]

    def __setitem__(self, val, value):
        self._q[val] = value

    def to_obj(self):
        return ('Q', self._q)

    @classmethod
    def from_obj(cls, o):
        name, _q = o 
        assert name == 'Q'
        q = Q()
        q._q = _q
        return q

    def max_a(self):
        # best next char for this state.
        c = self.chars[0]
        best = [c]
        maxq = self[c]
        for char in self.chars:
            q = self[char]
            if q > maxq:
               maxq, c = q, char
               best = [c]
            elif q == maxq:
                best.append(char)
        return random.choice(best)

    def __str__(self):
        return "q: %s" % str([(i, self._q[i]) for i in self._q if self._q[i] != 0])

    def __repr__(self):
        return str(self)


Alpha = 0.01 # Learning rate
Beta = 0.9    # Discounting factor

class QPolicy:
    def __init__(self):
        self._q, self._time_step = Q(), 0

    def q(self):
        return self._q

    def to_obj(self):
        return ('QPolicy', self._q.to_obj())

    @classmethod
    def from_obj(cls, o):
        name, obj = o
        assert name == 'QPolicy'
        qp = QPolicy()
        qp._q = Q.from_obj(obj)
        return qp

    def __str__(self):
        return "policy: %s" % self._q

    def __repr__(self):
        return str(self)

    def next_char(self):
        s = random.randint(0, self._time_step)
        self._time_step += 1
        if s == 0:
            return random.choice(All_Characters)
        else:
            return self._q.max_a()

    def max_a_val(self):
        a_char = self._q.max_a()
        return self._q[a_char]

    def update(self, a_char, last_max_q, reward):
        # Q(a,s)  = (1-alpha)*Q(a,s) + alpha(R(s) + beta*max_a(Q(a_,s_)))
        q_now = self._q[a_char]
        q_new = (1 - Alpha) * q_now + Alpha*(reward + Beta*last_max_q)
        self._q[a_char] = q_new

class Reward:
    Append = 0
    Trim = -1
    Complete = 100
    No = 0

class Executor:
    trainer = 'mjs_s'
    def __init__(self):
        self.pipeline = "./mjs -e '%s'"
        self.error = False

    def run(self, arg):
        exitcode, result = subprocess.getstatusoutput(self.pipeline % arg)
        if exitcode != 0 or result.startswith('MJS error: parse error at'):
            self.error = True
            if result.endswith('[]'):
                self.need_more = True
            elif result.endswith(']'):
                self.need_more = False
            else:
                assert False
        else:
            self.error = False
        return result

POLICY = '%s/policy.json' % Executor.trainer
TPOLICY = '%s.tmp' % POLICY

RESULTS = '%s/results.txt' % Executor.trainer
TRESULTS = '%s.tmp' % RESULTS



class Predictor:
    def __init__(self):
        self.load_policy()
        self.executor = Executor()

    def get_state(self, arg):
        skey = QState.get_key(arg)
        if skey not in self.states: self.states[skey] = QState(skey)
        return skey, self.states[skey]

    def dump_policy(self):
        d = []
        for key in self.states:
            d.append((key, self.states[key].to_obj()))
        json.dump(d, open(TPOLICY, 'w'))
        os.rename(TPOLICY, POLICY)

    def load_policy(self):
        self.states = {}
        if os.path.exists(POLICY):
            d = json.load(open(POLICY,'r'))
            for k,v in d:
                self.states[k] = QState.from_obj(v)

    def process(self, arg):
        for i in range(1, 100000):
            if len(arg) > 1000: break
            # First, get this state
            skey,state = self.get_state(arg)

            # get our next character
            c = state._policy.next_char()
            arg += c
            print("%d: %s" % (QState.Counter, repr(arg)))
            last_max_q = state._policy.max_a_val()

            # Now, see how it does.
            reward = Reward.No
            out = self.executor.run(arg)
            if self.executor.error:
                if self.executor.need_more:
                    #print('append')
                    reward = Reward.Append
                else:
                    #print('trim')
                    reward = Reward.Trim
                    arg = arg[:-1]
                state._policy.update(c, last_max_q, reward)
            else:
                if len(arg) > 10:
                    reward = min(len(arg)*2, Reward.Complete)
                    state._policy.update(c, last_max_q, reward)
                    self.dump_policy()
                    if os.path.exists(RESULTS):
                        shutil.copy(RESULTS, TRESULTS)
                    with open(TRESULTS, 'a') as f:
                        f.write(json.dumps({'string':arg}) + "\n")

                    os.rename(TRESULTS, RESULTS)
                    break
                else:
                    reward = Reward.Append
                    state._policy.update(c, last_max_q, reward)


def main(my_vars):
    p = Predictor()
    p.process(my_vars[1] if len(my_vars) > 1 else '')

main(sys.argv)
