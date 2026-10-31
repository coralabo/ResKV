# SPDX-License-Identifier: Apache-2.0
"""RULER string-match scoring without model or dataframe dependencies."""

import re

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f]")


def string_match_part(preds, refs) -> float:
    """RULER scoring (QA): partial match.

    preds: List[str]
    refs:  List[List[str]] (each sample can have multiple references)
    """
    score = (
        sum(
            max(1.0 if r.lower() in pred.lower() else 0.0 for r in ref)
            for pred, ref in zip(preds, refs, strict=True)
        )
        / len(preds)
        * 100
    )
    return round(score, 2)


def string_match_all(preds, refs) -> float:
    """RULER non-QA scoring: mean fraction of reference strings found."""
    score = (
        sum(
            sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) / len(ref)
            for pred, ref in zip(preds, refs, strict=True)
        )
        / len(preds)
        * 100
    )
    return round(score, 2)


def scorer(dataset, predictions, answers):
    """Score one task, preserving control-character cleaning and per-task rounding."""
    cleaned = [_CONTROL_CHARACTERS.sub("", text.strip()).strip() for text in predictions]
    metric = string_match_part if dataset.split("_")[0] == "qa" else string_match_all
    return metric(cleaned, answers)
