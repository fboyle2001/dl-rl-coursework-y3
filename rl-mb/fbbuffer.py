# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import numpy.random as npr
import torch
import os
import copy

from sortedcontainers import SortedSet

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.capacity = capacity
        self.device = device

        self.pixels = False
        self.empty_data()

        self.done_idxs = SortedSet()
        self.global_idx = 0
        self.global_last_save = 0

        self.normalize_obs = False

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del d['obses'], d['next_obses'], d['actions'], d['rewards'], \
          d['not_dones'], d['not_dones_no_max']
        return d

    def __setstate__(self, d):
        self.__dict__ = d

        # Manually need to re-load the transitions with load()
        self.empty_data()


    def empty_data(self):
        obs_dtype = np.float32 if not self.pixels else np.uint8
        obs_shape = self.obs_shape
        action_shape = self.action_shape
        capacity = self.capacity

        self.obses = np.empty((capacity, obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.payload = []
        self.done_idxs = None


    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        # For saving
        self.payload.append((
            obs.copy(), next_obs.copy(),
            action.copy(), reward,
            not done, not done_no_max
        ))

        # if self.full and not self.not_dones[self.idx]:
        if done:
            self.done_idxs.add(self.idx) # type: ignore
        elif self.full:
            self.done_idxs.discard(self.idx)  # type: ignore

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.global_idx += 1
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx,
            size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)



        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max


    def sample_multistep(self, batch_size, T):
        assert batch_size < self.idx or self.full

        last_idx = self.capacity if self.full else self.idx
        last_idx -= T

        # raw here means the "coalesced" indices that map to valid
        # indicies that are more than T steps away from a done
        done_idxs_sorted = np.array(list(self.done_idxs) + [last_idx])  # type: ignore
        n_done = len(done_idxs_sorted)
        done_idxs_raw = done_idxs_sorted - np.arange(1, n_done+1)*T

        samples_raw = npr.choice(
            last_idx-(T+1)*n_done, size=batch_size,
            replace=True # for speed
        )
        samples_raw = sorted(samples_raw)  # type: ignore
        js = np.searchsorted(done_idxs_raw, samples_raw)  # type: ignore
        offsets = done_idxs_raw[js] - samples_raw + T
        start_idxs = done_idxs_sorted[js] - offsets

        obses, actions, rewards = [], [], []

        for t in range(T):
            obses.append(self.obses[start_idxs + t])
            actions.append(self.actions[start_idxs + t])
            rewards.append(self.rewards[start_idxs + t])
            assert np.all(self.not_dones[start_idxs + t])

        obses = np.stack(obses)
        actions = np.stack(actions)
        rewards = np.stack(rewards).squeeze(2)

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)

        return obses, actions, rewards

    def save_data(self, save_dir):
        if self.global_idx == self.global_last_save:
            return
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = os.path.join(
            save_dir, f'{self.global_last_save:08d}_{self.global_idx:08d}.pt')

        payload = list(zip(*self.payload))
        payload = [np.vstack(x) for x in payload]
        self.global_last_save = self.global_idx
        torch.save(payload, path)
        self.payload = []


    def load_data(self, save_dir):
        def parse_chunk(chunk):
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            return (start, end)


        self.idx = 0

        chunks = os.listdir(save_dir)
        chunks = filter(lambda fname: 'stats' not in fname, chunks)
        chunks = sorted(chunks, key=lambda x: int(x.split('_')[0]))

        self.full = self.global_idx > self.capacity
        global_beginning = self.global_idx - self.capacity if self.full else 0

        for chunk in chunks:
            global_start, global_end = parse_chunk(chunk)
            if global_start >= self.global_idx:
                continue
            start = global_start - global_beginning
            end = global_end - global_beginning
            if end <= 0:
                continue

            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            if start < 0:
                payload = [x[-start:] for x in payload]
                start = 0
            assert self.idx == start

            obses = payload[0]
            next_obses = payload[1]

            self.obses[start:end] = obses
            self.next_obses[start:end] = next_obses
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.not_dones_no_max[start:end] = payload[5]
            self.idx = end

        self.last_save = self.idx

        if self.full:
            assert self.idx == self.capacity
            self.idx = 0

        last_idx = self.capacity if self.full else self.idx
        self.done_idxs = SortedSet(np.where(1.-self.not_dones[:last_idx])[0])