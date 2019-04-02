import copy
from functools import reduce
import itertools
import numpy as np
import logging
import pandas as pd
from math import factorial
from scipy.stats import nbinom
import combinatorics
import seaborn as sns
from matplotlib import pyplot as plt

from enum import Enum


class Events(Enum):
    bb = 0
    x1b = 1
    x2b = 2
    x3b = 3
    x4b = 4


EventMap = {x.value: x for x in Events}


def evolve_state(base_state, ev):
    new_state = copy.deepcopy(base_state)
    if ev == Events.bb:
        new_state[0] = 1
        new_state[1] = 1 if base_state[0] == 1 else base_state[1]
        new_state[2] = 1 if base_state[0] == 1 and base_state[
            1] == 1 else base_state[2]
    elif ev == Events.x1b:
        new_state[0] = 1
        new_state[1] = base_state[0]
        new_state[2] = base_state[1]
    elif ev == Events.x2b:
        new_state[0] = 0
        new_state[1] = 1
        new_state[2] = base_state[0]
    elif ev == Events.x3b:
        new_state[0] = 0
        new_state[1] = 0
        new_state[2] = 1
    elif ev == Events.x4b:
        new_state[0] = 0
        new_state[1] = 0
        new_state[2] = 0
    else:
        raise ValueError

    return new_state


def state_to_tuple(state):
    ans = [0] * len(Events)
    for v in state:
        idx = v.value
        ans[idx] += 1
    return tuple(ans)


def lob_from_seq(seq, base_state):
    if len(seq) == 0:
        return sum(base_state)
    else:
        ev = seq[0]
        return lob_from_seq(seq[1:], evolve_state(base_state, ev))


def unique_permutations(items):
    cache = set()
    it = itertools.permutations(items)
    for v in it:
        in_cache = v in cache
        #print(v, in_cache)
        cache.add(v)
        if not in_cache:
            #print(v, "yield")
            yield v
        else:
            #print(v, "not yield")
            pass


def get_all_seq(n):
    return set(
        list(
            itertools.permutations(
                [Events.bb, Events.x1b, Events.x2b, Events.x3b, Events.x4b
                 ] * 3, n)))


def seq_to_str(seq):
    return (''.join([str(e.value) for e in seq]) + '_' * (3 - len(seq)))[::-1]


def variance_decompose_c(prob_df):
    tmp = (prob_df
    .loc[:, ["runs", "pseq_combo", "pn", "pcombo", "pcombo_n", "prob", "total_combinations"]]
    .assign(z = lambda x: x.pseq_combo * x.runs, z2 = lambda x: x.pseq_combo * x.runs* x.runs)
    .groupby("total_combinations")
    .sum()
    .assign(w = lambda x: x.z * x.prob, w2 = lambda x: x.z * x.z * x.prob, 
    v =lambda x: x.z2 - x.z*x.z, ev=lambda x: x.prob*x.v)
    .sum()
    )
    return {"ev": tmp.ev, "ve": tmp.w2-tmp.w*tmp.w}


def variance_decompose(prob_df):
    tmp = (prob_df
    .loc[:, ["runs", "pn", "pcombo", "pcombo_n", "prob", "total_pa"]]
    .assign(z = lambda x: x.pcombo_n * x.runs, z2 = lambda x: x.pcombo_n * x.runs* x.runs)
    .groupby("total_pa")
    .sum()
    .assign(w = lambda x: x.z * x.prob, w2 = lambda x: x.z * x.z * x.prob, 
    v =lambda x: x.z2 - x.z*x.z, ev=lambda x: x.prob*x.v)
    .sum()
    )
    return {"ev": tmp.ev, "ve": tmp.w2-tmp.w*tmp.w}

class StateEnumerator:
    def __init__(self, max_pa, number_events=5):
        if max_pa > 23:
            raise ValueError("max_pa must be <= 23")
        self.max_pa = max_pa
        self.last_3 = self._last_three()
        self.number_events = number_events
        self.partition_combinations = None
        self._combinatorics_df = None

    @staticmethod
    def str_to_tuple(seq_str):
        count = [0] * 5
        for seq_char in [
                seq_char_ for seq_char_ in seq_str if seq_char_ != '_'
        ]:
            count[int(seq_char)] += 1
        return tuple(count)

    @staticmethod
    def _add_tuples(tuple1, tuple2):
        if len(tuple1) != len(tuple2):
            raise ValueError
        return tuple([tuple1[i] + tuple2[i] for i in range(len(tuple1))])

    @staticmethod
    def _merge_left_right_one(left_df, right_df):
        merged_df = pd.concat(
            [left_df.reset_index(drop=True),
             right_df.reset_index(drop=True)],
            axis=1,
            ignore_index=True)
        return merged_df

    @staticmethod
    def merge_left_right(base_df, right_df):
        nrow = len(base_df)
        dfs = []
        for i in range(nrow):
            if i % 1000 == 0:
                logging.info(print(r"#{} of {}".format(i, nrow)))
            row_df = pd.DataFrame(base_df.iloc[i, :]).T
            left_df = pd.concat([row_df] * len(right_df))
            merged_df = StateEnumerator._merge_left_right_one(
                left_df, right_df)
            dfs.append(merged_df)
        return pd.concat(dfs, axis=0)

    @property
    def combinatorics_df(self):
        if self._combinatorics_df is None:
            self._combinatorics_df = self._join_start_end()
        return self._combinatorics_df
    
    def _join_start_end(self):
        self.partition_combinations = self._all_combinations()
        # right_df = self.partition_combinations.loc[self.partition_combinations.pa_ == 0, :]
        # dfs = [self._merge_left_right(self._seq_runs_df(i)

        dfs = [
            self.merge_left_right(
                self.partition_combinations.loc[self.partition_combinations.pa_
                                                == 0, :], self._seq_runs_df(i))
            for i in range(3)
        ]

        df = pd.concat(dfs, axis=0, ignore_index=True).reset_index(drop=True)
        se_df = pd.concat([
            df,
            self.merge_left_right(self.partition_combinations, self.last_3)
        ],
                          axis=0,
                          ignore_index=True).reset_index(drop=True)
        se_df.columns = [
            "combinations", "pa_start", "multiplicity", "lob", "total_pa",
            "runs", "seq_str"
        ]

        se_df = se_df.assign(
            total_combinations=se_df.apply(
                lambda r: self._add_tuples(r.combinations,
                                           self.str_to_tuple(r.seq_str)),
                axis=1))

        x = se_df.total_pa + se_df.pa_start
        se_df = se_df.assign(total_pa=x)

        x = se_df.runs + se_df.pa_start
        se_df = se_df.assign(runs=x)
        return se_df

    def _seq_runs_df(self, seq_len):
        if seq_len > 3:
            logging.warning(
                "seq_len must be less than or equal to 3 not {}".format(
                    seq_len))
        res = []
        for i in range(seq_len, seq_len + 1):
            pa = i + 3
            all_seq = get_all_seq(min(i, 3))
            for seq_count, seq in enumerate(all_seq):
                seq_str = seq_to_str(seq)
                lob = lob_from_seq(seq, [0, 0, 0])
                logging.debug(seq_count, seq, seq_str, lob)
                res.append({
                    "lob": lob,
                    "pa": pa,
                    "runs": pa - 3 - lob,
                    "seq_str": seq_str
                })
        df = pd.DataFrame(res).sort_values("seq_str").reset_index().iloc[:, 1:]
        return df

    def _last_three(self):
        """
        enumerates all possibilities for the last 3 PA
        Returns: DataFrame

        """
        return self._seq_runs_df(seq_len=3)

    def _all_combinations(self):
        x = combinatorics.partition_combinations(self.max_pa,
                                                 self.number_events)
        dfs = []
        for i in range(self.max_pa):
            df = pd.DataFrame({
                "combination": sorted(list(x[i]), reverse=True),
                "pa_": i
            })
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        return df.assign(
            multiplicity=df.combination.apply(combinatorics.perm_number))

    def compute_probability(self, counts, probs):
        return np.product(
            [probs[i]**counts[i] for i in range(len(probs)) if counts[i] > 0])

    def get_prob_df(self, se_df, probs):

        sum_probs = sum(probs)

        if not (0 <= sum_probs < 1):
            raise ValueError
        if any(map(lambda p: p < 0 or p > 1, probs)):
            raise ValueError

        out_prob = 1 - sum_probs
        conditional_probs = list(map(lambda p: p / sum_probs, probs))
        nb_dist = nbinom(3, out_prob)
        prob_n_lookup = {x: nb_dist.pmf(x) for x in range(max(se_df.total_pa))}
        prob_df = se_df.assign(
            pn=se_df.total_pa.apply(lambda x: prob_n_lookup[x - 3]),
            pcombo=se_df.total_combinations.apply(
                lambda x: self.compute_probability(x, conditional_probs))
        ).assign(pcombo_n=lambda x: x.pcombo * x.multiplicity,
        prob=lambda x: x.pcombo_n * x.pn)
        
        normalize_df = prob_df.groupby(["total_combinations"]).multiplicity.sum()
        print(prob_df.iloc[0:3, :])
        print(normalize_df.iloc[0:2])
        prob_df = (
            pd.merge(prob_df, normalize_df, on="total_combinations")
            .rename(columns={"multiplicity_x": "multiplicity"})
            .assign(pseq_combo=lambda x: x.multiplicity / x.multiplicity_y).drop(columns=["multiplicity_y"])
        )

        int_columns = ["pa_start", "multiplicity", "lob", "total_pa", "runs"]
        float_columns = ["pn", "pcombo", "pcombo_n", "pseq_combo", "prob"]
        for col in int_columns:
            prob_df.loc[:, col] = pd.to_numeric(prob_df.loc[:, col])
        for col in float_columns:
            prob_df.loc[:, col] = pd.to_numeric(prob_df.loc[:, col])

        return prob_df

# runs given n
# prob_df.drop("combinations", axis=1).assign(z=prob_df.runs*prob_df.pcombo_n, z2=prob_df.runs*prob_df.runs*prob_df.pcombo_n).groupby("total_pa").sum()#
# 
# runs given combo
#  prob_df.drop("combinations", axis=1).assign(z=prob_df.runs*prob_df.pseq_combo, z2=prob_df.runs*prob_df.runs*prob_df.pseq_combo).groupby("total_combinations").sum()
