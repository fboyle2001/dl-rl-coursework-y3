import numpy as np
from collections import namedtuple

#@title Prioritised Replay Buffer

BufferSample = namedtuple("BufferSample", ["states", "actions", "rewards", "next_states", "terminals"])

class BinarySumTree:
    """
    **TODO: Cite the Prioritised Replay Buffer paper**
    https://www.fcodelabs.com/2019/03/18/Sum-Tree-Introduction/
    Data is stored in the following format [a, b, c, d, e, f, g] representing the tree:

         a
        / \
      b     c
     / \   / \
    d   e f   g 

    so for a parent node i it's left child is at 2i + 1 and it's right child is at 2i + 2

    A tree with N leaf nodes has a total of 2N - 1 total nodes

    The tree only stores the priorities, the replays are stored in the buffer which separates
    the sum tree from the replay buffer

    The purpose of the sum tree is that each parent is equal to the sum of its children
    e.g. b = d + e, c = f + g, a = b + c = d + e + f + g
    so the root node is the sum of the whole tree's values
    """

    def __init__(self, leaf_power):
        # leaf_count must be a power of 2
        # Nodes store the priorities
        self.leaf_power = leaf_power
        self.leaf_count = 2 ** leaf_power
        self.capacity = 2 * self.leaf_count - 1
        self.nodes = np.zeros(self.capacity)

    def _derive_real_leaf_index(self, leaf_index):
        # There are leaf_count - 1 parent nodes so offset the leaf index by this
        return leaf_index + self.leaf_count - 1

    def set_leaf(self, leaf_index, value):
        delta = value - self.get_leaf(leaf_index)
        self.update_leaf(leaf_index, delta)

    def update_leaf(self, leaf_index, delta):
        self._propagate_change(self._derive_real_leaf_index(leaf_index), delta)

    def _get_node(self, index):
        return self.nodes[index]

    def get_leaf(self, leaf_index):
        return self._get_node(self._derive_real_leaf_index(leaf_index))

    def _propagate_change(self, index, delta):
        # Propagation of an update takes O(log N) steps
        # Update the current index value which the change delta
        self.nodes[index] += delta

        # Reached the root then stop
        if index == 0:
            return

        # Propagate up the tree to the parent recursively
        # For child nodes they are positioned at either 2i + 1 or 2i + 2 where i is the parent
        # left: ((2i + 1) + 1) // 2 - 1 = i + 1 - 1 = i
        # right: ((2i + 2) + 1) // 2 - 1 = i + 1 - i = i
        self._propagate_change((index + 1) // 2 - 1, delta)

    def find_closest(self, value):
        # Start at the parent
        index = self._find_closest_index(value, 0)
        return index, self.get_leaf(index)

    def _find_closest_index(self, value, index):
        # leaf node
        if index >= self.leaf_count - 1:
            return index - (self.leaf_count - 1)

        left_index = 2 * index + 1
        right_index = left_index + 1

        left_value = self._get_node(left_index)
        right_value = self._get_node(right_index)

        # value is less than the sum of the children of this subtree
        # so search in here
        if value <= left_value:
            return self._find_closest_index(value, left_index)

        # otherwise try the right subtree
        return self._find_closest_index(value - left_value, right_index)

    def _get_leaves(self):
        return self.nodes[self.leaf_count - 1:]

    def _recompute_entire_tree(self):
        duplicated_tree = BinarySumTree(self.leaf_power)
        copied_leaves = self._get_leaves().copy()
        
        for i, value in enumerate(copied_leaves):
            duplicated_tree.set_leaf(i, value)

        self.nodes = duplicated_tree.nodes

    @property
    def sum_total(self):
        # The parent will be equal to the sum of all leaves
        return self._get_node(0)

    def get_max_leaf_value(self, limit):
        return np.max(self._get_leaves()[:limit])
  
    def get_min_leaf_value(self, limit):
        return np.min(self._get_leaves()[:limit])

class PrioritisedReplayBuffer:
    def __init__(self, state_dim, action_dim, leaf_power, device):
        self.device = device

        # Priority replay buffer
        self.priorities = BinarySumTree(leaf_power)
        self.alpha = 0.6
        self.beta = 0.4 # --> 1
        self.epsilon = 1e-3
        self.delta = 0
        self.beta_increment = 1e-3 #6.25e-5 # unsure about this value
        self.td_error_clip = 1 # Section 4 clip td error to [-1, 1] (or [epsilon, 1] since |error + epsilon| is used)

        self.max_size = self.priorities.leaf_count

        # Buffers
        self.states      = np.zeros((self.max_size, state_dim))
        self.actions     = np.zeros((self.max_size, action_dim))
        self.rewards     = np.zeros(self.max_size)
        self.next_states = np.zeros((self.max_size, state_dim))
        self.terminals   = np.zeros(self.max_size, dtype="bool")

        # Buffer state information
        self.pointer = 0
        self.count = 0

    def store_replay(self, state, action, reward, next_state, is_terminal, td_error):
        # print(f"Storing replay at {self.pointer}")
        self.states[self.pointer]      = state
        self.actions[self.pointer]     = action
        self.rewards[self.pointer]     = reward
        self.next_states[self.pointer] = next_state
        self.terminals[self.pointer]   = not is_terminal
        
        # Set the priority in the Binary Sum Tree
        if td_error is None:
            priority = self.priorities.get_max_leaf_value(self.count) if self.count != 0 else 1
            # print(f"Setting priority to the max which is {self.priorities.max_leaf_value} actually using {priority}")
        else:
            # Power should be fine outside the min since they are clipped 
            # at epsilon <= p <= 1 so 0 <= p ** a <= 1 since 0 <= alpha <= 1
            priority = np.min([np.abs(td_error) + epsilon, self.td_error_clip]) ** self.alpha

        self.priorities.set_leaf(self.pointer, priority)
        # print(f"Set leaf {self.pointer} to priority {priority}")

        self.count = min(self.count + 1, self.max_size)
        self.pointer = (self.pointer + 1) % self.max_size

    def update_priorities(self, indexes, td_errors):
        # Element-wise minimum
        td_errors = np.minimum(td_errors + self.epsilon, self.td_error_clip) ** self.alpha

        for td_index in range(len(td_errors)):
            self.priorities.set_leaf(indexes[td_index], td_errors[td_index])

    def sample_buffer(self, k):
        # Don't need to worry about empty slots since the priorities intrinsically handle it

        """
        initialise empty minibatch of size k, indexes and weights
        divide total priority by k so we have [0, t/k], [t/k, 2t/k], ... blocks to sample from
        set beta = min(1, beta + step_size) 
        calculate importance sampling normalisation weight

        for i = 0...k:
            priority ~ Uniform(i * t / k, (i + 1) * t / k)
            get the nearest index and priority from the tree
            calculate sample probability as priority / total_priority
            calculate importance sampling weight
            add the replay to the minibatch
        """

        self.beta = np.min([self.beta + self.beta_increment, 1])

        indexes = []
        importance_sampling_weights = []
        
        # Partition uniformly
        segment_range = self.priorities.sum_total / k
        # Normalise by the maximum weight 1 / x -> inf as x -> 0 so use min_leaf_value
        normalisation_weight = (1 / (self.count * self.priorities.get_min_leaf_value(self.count))) ** self.beta 

        # print(f"Normalisation weight (min leaf={self.priorities.get_min_leaf_value(self.count)}) {normalisation_weight}")

        # Sample once from each partition
        for i in range(k):
            lower_bound = i * segment_range
            upper_bound = (i + 1) * segment_range

            # Find the nearest index to the given priority
            approx_priority = np.random.uniform(lower_bound, upper_bound)
            index, priority = self.priorities.find_closest(approx_priority)

            # Probability of sampling this replay
            probability = priority / self.priorities.sum_total

            # Importance sampling weight
            weight = ((1 / (self.count * probability)) ** self.beta) / normalisation_weight

            # print(f"Priority {priority} TP {self.priorities.sum_total} Probability {probability} weight {weight} prod {self.count * probability}")

            importance_sampling_weights.append(weight)
            indexes.append(index)

        indexes = np.array(indexes)
        # print(f"Used indexes: (current max is {self.count}, pointer is at {self.pointer})")
        # print(indexes)
        # print(self.priorities._get_leaves()[indexes])

        importance_sampling_weights = np.array(importance_sampling_weights)

        if np.any(np.isnan(importance_sampling_weights)) or np.any(np.isinf(importance_sampling_weights)):
            print(f"NAN: {np.any(np.isnan(importance_sampling_weights))}, INF: {np.any(np.isinf(importance_sampling_weights))}")
            print(f"current max is {self.count}, pointer is at {self.pointer}")
            print(f"Total priority is {self.priorities.sum_total}")
            print(f"Normalisation weight is {normalisation_weight}")
            print("Indexes")
            print(indexes)
            print("ISWs")
            print(importance_sampling_weights)
            print("Priorities")
            print(self.priorities._get_leaves()[indexes])

            r_time = time.time()

            np.save(f"tree_nodes_{r_time}.npy", self.priorities.nodes)
            np.save(f"indexes_{r_time}.npy", indexes)
            np.save(f"isws_{r_time}.npy", importance_sampling_weights)
            np.save(f"prios_{r_time}.npy", self.priorities._get_leaves()[indexes])

            print("Saved tree array, indexes, ISWs and priorities")

        importance_sampling_weights = torch.tensor(importance_sampling_weights).to(self.device)

        buffer = BufferSample(
            torch.FloatTensor(self.states[indexes]).to(self.device),
            torch.FloatTensor(self.actions[indexes]).to(self.device),
            torch.FloatTensor(self.rewards[indexes]).to(self.device),
            torch.FloatTensor(self.next_states[indexes]).to(self.device),
            torch.FloatTensor(self.terminals[indexes]).to(self.device)
        )

        return buffer, indexes, importance_sampling_weights

    @property
    def current_buffer_size(self):
        return self.count

preloaded_nodes = np.load("tree_nodes_1642525379.3858314.npy")
indexes = np.load("indexes_1642525379.3858314.npy")

# print(indexes[-1])

tree = BinarySumTree(20)
tree.nodes = preloaded_nodes

# lb = 255 * tree.sum_total / 256
# ub = 256 * tree.sum_total / 256

# approx_priority = np.random.uniform(lb, ub - 1)

# print(approx_priority, lb, ub)

# index, priority = tree.find_closest(ub)

# print(index, priority)

print(tree.sum_total, np.sum(tree._get_leaves()))
index, priority = tree.find_closest(tree.sum_total)
print(index, priority)

tree._recompute_entire_tree()

print(tree.sum_total, np.sum(tree._get_leaves()))
index, priority = tree.find_closest(tree.sum_total)
print(index, priority)

# print(tree._get_leaves()[indexes])

    
