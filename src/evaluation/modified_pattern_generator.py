import ast
from dataclasses import dataclass
from os import path
from pathlib import Path

import pandas as pd

from src.data_generation.generate_synthetic_segmented_dataset import calculate_mae
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns, check_valid_coefficients, \
    PatternCols
from src.utils.configurations import MODIFIED_PATTERNS_PATH


@dataclass
class ModifiedCols:
    id = PatternCols.id
    relaxed_pattern = PatternCols.relaxed_patterns
    modified_pattern = 'Modified Pattern'
    mae = 'MAE'


class ModifiedPatternGenerator:
    """
    Generates the patterns that are by mae moved towards the inside of the elliptope.
    This creates reference group of patterns for the L_0 = 0 identity
    test. For each relaxed canonical pattern, produces a single deterministic
    pattern obtained by shifting every coefficient away from its current extreme
    value by mae.
    """

    def __init__(self, mae: float):
        """
        :param mae: amount to move each coefficient away from its current extreme value.
        """
        model_canonical_patterns = ModelCorrelationPatterns()
        self.relaxed_patterns = model_canonical_patterns.relaxed_patterns()
        self.mae = mae

    def __get_sign_modifier(self, coeff: float, pattern: list) -> float:
        # sign will be the same as the sign of the sum of the pattern
        if coeff == 0:
            total = sum(pattern)
            return self.mae if total <= 0 else -self.mae
        else:
            return self.mae if coeff < 0 else -self.mae

    def __shift_pattern_by_mae(self, pattern: list) -> list:
        """
        Shifts a pattern by mae away from the elliptope boundaries. MAE
        is added if the sum of the coefficients is negative, respectively
        subtracted if the sum is positive or zero
        :param pattern: list of coefficients
        :return: modified list of coefficients
        """
        return [round(coeff + self.__get_sign_modifier(coeff, pattern), 2) for coeff in pattern]

    def generate(self) -> pd.DataFrame:
        """
        :return: pd.DataFrame with columns:
            BoundaryPatternCols.pattern_index -> canonical pattern index
            BoundaryPatternCols.relaxed_pattern -> original relaxed pattern
            BoundaryPatternCols.boundary_pattern -> pattern shifted to the MAE boundary
            BoundaryPatternCols.mae_boundary -> mae_boundary used, for traceability
            BoundaryPatternCols.is_valid -> PSD check result, so invalid rows are
                visible in the file rather than silently produced
        """
        rows = []
        for pattern_index, pattern in self.relaxed_patterns.items():
            modified_pattern = self.__shift_pattern_by_mae(pattern)
            assert check_valid_coefficients(
                *modified_pattern), f'Invalid modified pattern: {modified_pattern}, for pattern: {pattern}'
            rows.append({
                ModifiedCols.id: pattern_index,
                ModifiedCols.relaxed_pattern: pattern,
                ModifiedCols.modified_pattern: modified_pattern,
                ModifiedCols.mae: calculate_mae(modified_pattern, pattern, round_to=2),
            })
        return pd.DataFrame(rows)


def save_modified_patterns(df: pd.DataFrame, file_path: str = MODIFIED_PATTERNS_PATH):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


def read_modified_patterns(file: str = MODIFIED_PATTERNS_PATH) -> pd.DataFrame:
    assert (path.exists(file))
    df = pd.read_csv(file)
    # change to list type
    df[ModifiedCols.relaxed_pattern] = df[ModifiedCols.relaxed_pattern].apply(lambda x: ast.literal_eval(x))
    df[ModifiedCols.modified_pattern] = df[ModifiedCols.modified_pattern].apply(lambda x: ast.literal_eval(x))
    return df


if __name__ == "__main__":
    "Generate modified patterns that are exactly MAE 0.1 towards the inside of the elliptope from the original pattern"
    generator = ModifiedPatternGenerator(mae=0.1)
    modified_patterns = generator.generate()

    save_modified_patterns(modified_patterns)
