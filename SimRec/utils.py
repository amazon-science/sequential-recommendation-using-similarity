import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from enum import auto, Enum
from itertools import chain

class LinearScheduleWithWarmup:
    def __init__(self, lambd, warmup_steps, lamb_steps):
        self.lambd = 0
        self.warmup_steps = warmup_steps
        self.lamb_steps = lamb_steps
        self.warmup_alpha = lambd / warmup_steps
        self.alpha = lambd / (warmup_steps - lamb_steps)
        self.bias = lambd * (1 - (warmup_steps / (warmup_steps - lamb_steps)))
        self.current_step = -1
        self.step()

    def get_lambd(self):
        return max(self.lambd, 0)

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            self.lambd = self.warmup_alpha * self.current_step
        else:
            self.lambd = self.alpha * self.current_step + self.bias

class NoneSchedule:
    def __init__(self, lambd):
        self.lambd = lambd

    def get_lambd(self):
        return self.lambd

    def step(self):
        pass

PAD_IDX = 0

def create_similarity_distirbution(similarity_indices, similarity_values, temperature, positive_indices):
    num_items = similarity_indices.shape[0]
    num_positives = positive_indices.shape[0]
    # (num_positives, top_k_similar)
    pos_similarity_indices =  torch.index_select(similarity_indices, index=positive_indices, dim=0)
    pos_similarity_values = torch.index_select(similarity_values, index=positive_indices, dim=0)
    
    # (num_positives, num_items)
    similarities = torch.full((num_positives, num_items), fill_value=-float('inf'), device=similarity_indices.device)
    similarities.scatter_(dim=1, index=pos_similarity_indices, src=pos_similarity_values)

    similarities /= temperature

    distribution = torch.nn.functional.softmax(similarities, dim=-1)
    return distribution

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return user, seq, pos, neg

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname, augmentations_fname=None):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    with open(fname, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
    if augmentations_fname is not None:
        with open(f'data/{augmentations_fname}.txt', 'r') as f:
            for line in f:
                u, i = line.rstrip().split(' ')
                u = int(u)
                i = int(i)
                usernum = max(u, usernum)
                itemnum = max(i, itemnum)
                User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# evaluate on test set
def evaluate_test(model, dataset, args):

    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    id_hr = defaultdict(list)
    id_ndcg = defaultdict(list)
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < 10:
            ndcg = 1 / np.log2(rank + 2)
            NDCG += ndcg
            HT += 1            
            id_hr[item_idx[0]].append(1)
            id_ndcg[item_idx[0]].append(ndcg)
        else:
            id_hr[item_idx[0]].append(0)
            id_ndcg[item_idx[0]].append(0)
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    return (NDCG / valid_user, HT / valid_user), id_hr, id_ndcg


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    id_hr = defaultdict(list)
    id_ndcg = defaultdict(list)
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < 10:
            ndcg = 1 / np.log2(rank + 2)
            NDCG += ndcg
            HT += 1
            id_hr[item_idx[0]].append(1)
            id_ndcg[item_idx[0]].append(ndcg)
        else:
            id_hr[item_idx[0]].append(0)
            id_ndcg[item_idx[0]].append(0)
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    return (NDCG / valid_user, HT / valid_user), id_hr, id_ndcg

# evaluate on train set
def evaluate_train(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    id_hr = defaultdict(list)
    id_ndcg = defaultdict(list)
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u][:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u][:-1])
        rated.add(0)
        item_idx = [train[u][-1]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < 10:
            ndcg = 1 / np.log2(rank + 2)
            NDCG += ndcg
            HT += 1
            id_hr[item_idx[0]].append(1)
            id_ndcg[item_idx[0]].append(ndcg)
        else:
            id_hr[item_idx[0]].append(0)
            id_ndcg[item_idx[0]].append(0)
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    return (NDCG / valid_user, HT / valid_user), id_hr, id_ndcg