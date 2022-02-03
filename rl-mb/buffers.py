from random import random
from turtle import done
from typing import Tuple, Any, Optional, Union
import time
import abc

import numpy as np
import torch

from collections import namedtuple

BufferSample = namedtuple("BufferSample", ["states", "actions", "rewards", "next_states", "terminals"])
Trajectory = namedtuple("Trajectory", ["states", "actions", "rewards", "next_states", "terminals"])

class AbstractReplayBuffer(abc.ABC):
    def __init__(self, state_dim: int, action_dim: int, max_size: int, device: Union[str, torch.device]):
        self.device = device

        self._state_dim = state_dim
        self._action_dim = action_dim

        # Buffers
        self.states       = np.zeros((max_size, state_dim))
        self.actions      = np.zeros((max_size, action_dim))
        self.rewards      = np.zeros(max_size)
        self.next_states  = np.zeros((max_size, state_dim))
        self.terminals    = np.zeros(max_size, dtype="bool")
        # Will store the end index of the trajectory and the length of the trajectory
        self.trajectories = np.zeros((max_size, 1))

        self.max_size = max_size
        self.pointer = 0
        self.count = 0
        self.traj_length = 0

    def is_full(self) -> bool:
        return self.count == self.max_size

    def store_replay(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, is_terminal: bool) -> None:
        self.traj_length += 1

        self.states[self.pointer]      = state
        self.actions[self.pointer]     = action
        self.rewards[self.pointer]     = reward
        self.next_states[self.pointer] = next_state
        self.terminals[self.pointer]   = not is_terminal
        self.trajectories[self.pointer] = self.traj_length if is_terminal else 0
        
        if is_terminal:
            self.traj_length = 0

        self.count = min(self.count + 1, self.max_size)
        self.pointer = (self.pointer + 1) % self.max_size

    def store_replays(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, is_terminals: np.ndarray) -> None:
        for replay_index in range(states.shape[0]):
            self.store_replay(states[replay_index], actions[replay_index], rewards[replay_index], next_states[replay_index], is_terminals[replay_index])

    def get_all_replays(self) -> BufferSample:
        buffer = BufferSample(
            torch.FloatTensor(self.states[:self.count]).to(self.device),
            torch.FloatTensor(self.actions[:self.count]).to(self.device),
            torch.FloatTensor(self.rewards[:self.count]).to(self.device),
            torch.FloatTensor(self.next_states[:self.count]).to(self.device),
            torch.FloatTensor(self.terminals[:self.count]).to(self.device)
        )

        return buffer

    @abc.abstractmethod
    def sample_buffer(self, batch_size: int) -> BufferSample:
        pass

    @abc.abstractmethod
    def sample_trajectories(self, trajectory_count: int, steps_per_trajectory: int):
        pass

class BinarySumTree:
    """
    **TODO: Cite the Prioritised Replay Buffer paper**
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
        self.error_tolerance = 0.1

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
        error = value - self.sum_total

        if self.error_tolerance < error < self.error_tolerance * 2:
            print(f"Error has exceeded limits {error} > {self.error_tolerance}")
            print(f"Known total is {self.sum_total}, real total is {np.sum(self._get_leaves())}")
            print("Recomputing the tree...")
            self._recompute_tree()
            print(f"Tree recomputed known total is {self.sum_total}, real total is {np.sum(self._get_leaves())}")
        elif np.abs(error) < self.error_tolerance:
            print(f"Encountered tolerance issue of {error}")
            value = self.sum_total - 10 * self.error_tolerance

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
        if value < left_value:
            return self._find_closest_index(value, left_index)

        # otherwise try the right subtree
        return self._find_closest_index(value - left_value, right_index)

    def _get_leaves(self):
        return self.nodes[self.leaf_count - 1:]

    def _recompute_tree(self):
        """
        Since we are constantly propagating differences between numbers up the tree
        over time error will accumulate in the parent S. 
        
        This is problematic when searching the tree for a value x satisfying:

          (x - S) < 0 

        since we will be searching outside the range of the sum tree and it will
        return an incorrect value.

        If (x - S) < E for some error tolerance E that we should recompute the tree
        completely which takes fixed time but is fairly costly.
        """
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

class StandardReplayBuffer(AbstractReplayBuffer):
    def __init__(self, state_dim: int, action_dim: int, max_size: int, device: Union[str, torch.device]):
        super(StandardReplayBuffer, self).__init__(state_dim, action_dim, max_size, device)

    def sample_buffer(self, batch_size: int) -> BufferSample:
        random_sample = np.random.choice(self.count, size=batch_size)

        return BufferSample(
            torch.FloatTensor(self.states[random_sample]).to(self.device),
            torch.FloatTensor(self.actions[random_sample]).to(self.device),
            torch.FloatTensor(self.rewards[random_sample]).to(self.device),
            torch.FloatTensor(self.next_states[random_sample]).to(self.device),
            torch.FloatTensor(self.terminals[random_sample]).to(self.device)
        )

    def sample_trajectories(self, trajectory_count: int, steps_per_trajectory: int):
        suitable_ends = np.where(self.trajectories[:, 0] > steps_per_trajectory)[0]
        selected_ends = np.random.choice(suitable_ends, trajectory_count, replace=True)
        trajectories = []

        for trajectory_end in selected_ends:
            trajectory_start = trajectory_end - steps_per_trajectory
            trajectory = torch.tensor(
                [
                    self.states[trajectory_start : trajectory_end],
                    self.actions[trajectory_start : trajectory_end],
                    self.rewards[trajectory_start : trajectory_end],
                    self.next_states[trajectory_start : trajectory_end],
                    self.terminals[trajectory_start : trajectory_end]
                ]
            ).to(self.device)

            trajectories.append(trajectory)

        return torch.stack(trajectories)

class PriorityReplayBuffer(AbstractReplayBuffer):
    def __init__(self, state_dim: int, action_dim: int, leaf_power: int, device: Union[str, torch.device]):
        super().__init__(state_dim, action_dim, 2 ** leaf_power, device)

        self.priorities = BinarySumTree(leaf_power)
        self.alpha = 0.6
        self.beta = 0.4 # --> 1
        self.epsilon = 1e-3
        self.delta = 0
        self.beta_increment = 1e-3 #6.25e-5 # unsure about this value
        self.td_error_clip = 1 # Section 4 clip td error to [-1, 1] (or [epsilon, 1] since |error + epsilon| is used)

    def store_replay(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, is_terminal: bool, td_error: Optional[float]) -> None:
        # Set the priority in the Binary Sum Tree
        if td_error is None:
            priority = self.priorities.get_max_leaf_value(self.count) if self.count != 0 else 1
            # print(f"Setting priority to the max which is {self.priorities.max_leaf_value} actually using {priority}")
        else:
            # Power should be fine outside the min since they are clipped 
            # at epsilon <= p <= 1 so 0 <= p ** a <= 1 since 0 <= alpha <= 1
            priority = np.min([np.abs(td_error) + self.epsilon, self.td_error_clip]) ** self.alpha

        self.priorities.set_leaf(self.pointer, priority)
        super().store_replay(state, action, reward, next_state, is_terminal)

    def update_priorities(self, indexes, td_errors):
        # Element-wise minimum
        td_errors = np.minimum(td_errors + self.epsilon, self.td_error_clip) ** self.alpha

        for td_index in range(len(td_errors)):
            self.priorities.set_leaf(indexes[td_index], td_errors[td_index])

    def sample_buffer(self, batch_size: int) -> Tuple[BufferSample, np.ndarray, torch.Tensor]:
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
        segment_range = self.priorities.sum_total / batch_size
        # Normalise by the maximum weight 1 / x -> inf as x -> 0 so use min_leaf_value
        normalisation_weight = (1 / (self.count * self.priorities.get_min_leaf_value(self.count))) ** self.beta 

        # print(f"Normalisation weight (min leaf={self.priorities.get_min_leaf_value(self.count)}) {normalisation_weight}")

        # Sample once from each partition
        for i in range(batch_size):
            lower_bound = i * segment_range
            upper_bound = (i + 1) * segment_range

            if i == batch_size - 1:
              upper_bound *= 0.99
              upper_bound =- 1

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
    
    def sample_trajectories(self, trajectory_count: int, steps_per_trajectory: int):
        raise NotImplementedError("Trajectories are unsupported for this buffer")

# https://pswww.slac.stanford.edu/svn-readonly/psdmrepo/RunSummary/trunk/src/welford.py
class Welford(object):
    """Knuth implementation of Welford algorithm.
    """

    def __init__(self, x=None):
        self._K = np.float64(0.)
        self.n = np.float64(0.)
        self._Ex = np.float64(0.)
        self._Ex2 = np.float64(0.)
        self.shape = None
        self._min = None
        self._max = None
        self._init = False
        self.__call__(x)

    def add_data(self, x):
        """Add data.
        """
        if x is None:
            return

        x = np.array(x)
        self.n += 1.
        if not self._init:
            self._init = True
            self._K = x
            self._min = x
            self._max = x
            self.shape = x.shape
        else:
            self._min = np.minimum(self._min, x)#  type: ignore
            self._max = np.maximum(self._max, x)#  type: ignore

        self._Ex += (x - self._K) / self.n
        self._Ex2 += (x - self._K) * (x - self._Ex)
        self._K = self._Ex

    def __call__(self, x):
        self.add_data(x)

    def max(self):
        """Max value for each element in array.
        """
        return self._max

    def min(self):
        """Min value for each element in array.
        """
        return self._min

    def mean(self, axis=None):
        """Compute the mean of accumulated data.
           Parameters
           ----------
           axis: None or int or tuple of ints, optional
                Axis or axes along which the means are computed. The default is to
                compute the mean of the flattened array.
        """
        if self.n < 1:
            return None

        val = np.array(self._K + self._Ex / np.float64(self.n))
        if axis:
            return val.mean(axis=axis)
        else:
            return val

    def sum(self, axis=None):
        """Compute the sum of accumulated data.
        """
        return self.mean(axis=axis)*self.n # type: ignore

    def var(self):
        """Compute the variance of accumulated data.
        """
        if self.n <= 1:
            return  np.zeros(self.shape) #  type: ignore

        val = np.array((self._Ex2 - (self._Ex*self._Ex)/np.float64(self.n)) / np.float64(self.n-1.))

        return val

    def std(self):
        """Compute the standard deviation of accumulated data.
        """
        return np.sqrt(self.var())

#    def __add__(self, val):
#        """Add two Welford objects.
#        """
#

    def __str__(self):
        if self._init:
            return "{} +- {}".format(self.mean(), self.std())
        else:
            return "{}".format(self.shape)

    def __repr__(self):
        return "< Welford: {:} >".format(str(self))

class MultiStepReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int, device: Union[str, torch.device], normalise: bool = False):
        self.device = device

        self._state_dim = state_dim
        self._action_dim = action_dim

        # Buffers
        self.states         = np.zeros((max_size, state_dim))
        self.actions        = np.zeros((max_size, action_dim))
        self.rewards        = np.zeros(max_size)
        self.next_states    = np.zeros((max_size, state_dim))
        self.true_not_dones = np.zeros(max_size, dtype="bool")
        self.not_dones      = np.zeros(max_size, dtype="bool")

        # Either stores 0 or a positive number at each index a positive number
        # indicates that it is the end of sequence, the number indicates how long the sequence is
        self.sequences      = np.zeros((max_size, 1))

        self.max_size = max_size
        self.pointer = 0
        self.count = 0
        self.sequence_length = 0
        self.welford = Welford()
        self.normalise = normalise

    def store(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, true_done: bool, done: bool) -> None:
        self.sequence_length += 1

        if self.normalise:
            # Lets normalise everything as it comes in
            self.welford.add_data(state)

        true_not_done = not true_done
        not_done = not done

        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.true_not_dones[self.pointer] = true_not_done
        self.not_dones[self.pointer] = not_done
        self.sequences[self.pointer] = self.sequence_length

        if done:
            self.sequence_length = 0

        self.count = min(self.count + 1, self.max_size)
        self.pointer = (self.pointer + 1) % self.max_size

    def get_obs_stats(self):
        MIN_STD = 1e-1
        MAX_STD = 10
        mean = self.welford.mean()
        std = self.welford.std()
        std[std < MIN_STD] = MIN_STD
        std[std > MAX_STD] = MAX_STD
        return mean, std

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        random_sample = np.random.choice(self.count, size=batch_size)
        states = self.states[random_sample]
        next_states = self.next_states[random_sample]

        if self.normalise:
            mu, sigma = self.get_obs_stats()
            states = (states - mu) / sigma
            next_states = (next_states - mu) / sigma

        return (
            torch.tensor(states, dtype=torch.float32).to(self.device),
            torch.tensor(self.actions[random_sample], dtype=torch.float32).to(self.device),
            torch.tensor(self.rewards[random_sample], dtype=torch.float32).to(self.device),
            torch.tensor(next_states, dtype=torch.float32).to(self.device),
            torch.tensor(self.true_not_dones[random_sample], dtype=torch.float32).to(self.device),
            torch.tensor(self.not_dones[random_sample], dtype=torch.float32).to(self.device)
        )

    def sample_sequences(self, sequences: int, sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Find the indices of the final replay in each sequence of min sequence_length
        final_replay_indices = np.where(self.sequences[:, 0] > sequence_length)[0]

        # Now select a random sample (sequences) of these
        selected_final_indices = np.random.choice(final_replay_indices, size=sequences, replace=True)

        # Sorted by timestep rather than sequence
        timestepped_states = []
        timestepped_actions = []
        timestepped_rewards = []

        mu, sigma = self.get_obs_stats() if self.normalise else(0, 1)

        for offset in range(0, sequence_length):
            time_states = (self.states[selected_final_indices - sequence_length + offset] - mu) / sigma
            timestepped_states.append(time_states)
            time_actions = self.actions[selected_final_indices - sequence_length + offset]
            timestepped_actions.append(time_actions)
            time_rewards = self.rewards[selected_final_indices - sequence_length + offset]
            timestepped_rewards.append(time_rewards)

        timestepped_states = np.stack(timestepped_states)
        timestepped_actions = np.stack(timestepped_actions)
        timestepped_rewards = np.stack(timestepped_rewards)

        return (
            torch.tensor(timestepped_states, dtype=torch.float32).to(self.device),
            torch.tensor(timestepped_actions, dtype=torch.float32).to(self.device),
            torch.tensor(timestepped_rewards, dtype=torch.float32).to(self.device)
        )

        # Shape is [sequence_length, sequences, ...] do whatever with these
