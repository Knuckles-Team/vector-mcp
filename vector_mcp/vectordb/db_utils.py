#!/usr/bin/python
# coding: utf-8
from agent_utilities import get_logger
from .base import QueryResults
from typing import Any

logger = get_logger(__name__)


def filter_results_by_distance(
    results: QueryResults, distance_threshold: float = -1
) -> QueryResults:
    """Filters results based on a distance threshold.

    Args:
        results: QueryResults | The query results. List[List[Tuple[Document, float]]]
        distance_threshold: The maximum distance allowed for results.

    Returns:
        QueryResults | A filtered results containing only distances smaller than the threshold.
    """
    if distance_threshold > 0:
        results = [
            [(key, value) for key, value in data if value < distance_threshold]
            for data in results
        ]

    return results


def chroma_results_to_query_results(
    data_dict: dict[str, list[list[Any]]], special_key="distances"
) -> QueryResults:
    """Converts a dictionary with list-of-list values to a list of tuples."""
    keys = [
        key
        for key in data_dict
        if key != special_key
        and data_dict[key] is not None
        and isinstance(data_dict[key][0], list)
    ]
    result = []
    data_special_key = data_dict[special_key]

    for i in range(len(data_special_key)):
        sub_result = []
        for j, distance in enumerate(data_special_key[i]):
            sub_dict = {}
            for key in keys:
                if len(data_dict[key]) > i:
                    sub_dict[key[:-1]] = data_dict[key][i][j]
            sub_result.append((sub_dict, distance))
        result.append(sub_result)

    return result
