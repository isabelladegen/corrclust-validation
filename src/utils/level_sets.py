import itertools

import numpy as np

from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.utils.distance_measures import l1_distance_from_matrices


class LevelSets:
    def __init__(self):
        # key is level, value is list of tuples of pairs
        self.pairs_for_level_lookup = self.__find_all_level_sets()
        self.levels = list(self.pairs_for_level_lookup.keys())
        # tuples of adjacent indices
        self.adjacent_indices = [(self.levels[i], self.levels[i + 1]) for i in range(len(self.levels) - 1)]
        # key is pair and value is level of that pair
        self.level_for_pair_lookup = {}
        for key, pattern_tuples_list in self.pairs_for_level_lookup.items():
            for pattern_pairs in pattern_tuples_list:
                self.level_for_pair_lookup[pattern_pairs] = key

    @staticmethod
    def __find_all_level_sets():
        """
        Calculates all possible level sets and which pattern tuples belong to it. This uses the relaxed pattern
        note patterns are added in both order to not need to worry how to look up so there is double the pairs in
        the dictionary
        :return: dictionary of key level set id and values list of tuples of pattern pairs in that level set
        """
        model_correlation_patterns = ModelCorrelationPatterns()
        # key pattern id value relaxed coefficient list
        relaxed_patterns = model_correlation_patterns.relaxed_patterns()

        # all combinations of patterns
        all_pattern_combinations = list(itertools.combinations_with_replacement(relaxed_patterns.keys(), 2))

        # dictionary of pattern id and tuples of pairs in that set
        level_sets = {i: [] for i in range(6)}

        # cycle through all pattern pairs and put them in the right level set based on L1 distance between pair
        for pair in all_pattern_combinations:
            p1 = relaxed_patterns[pair[0]]
            p2 = relaxed_patterns[pair[1]]

            # calculate l1 distance
            d = l1_distance_from_matrices(p1, p2)
            level = int(np.floor(d + 0.5))
            # add both ways round to always succeed lookup
            level_sets[level].append((pair[0], pair[1]))
            if pair[0] != pair[1]:
                level_sets[level].append((pair[1], pair[0]))

        # drop empty values
        clean_dict = {k: v for k, v in level_sets.items() if v}

        return clean_dict
