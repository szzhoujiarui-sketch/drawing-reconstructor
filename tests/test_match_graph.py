import numpy as np

from drawing_reconstructor.match_graph import MatchGraph
from drawing_reconstructor.models import MatchEdge


def edge(source, target, confidence):
    homography = np.eye(3, dtype=np.float64)
    homography[0, 2] = source - target
    return MatchEdge(
        source=source,
        target=target,
        matches=20,
        inliers=15,
        inlier_ratio=0.75,
        reprojection_error=0.5,
        confidence=confidence,
        homography=homography,
    )


def test_match_graph_prefers_higher_confidence_path():
    graph = MatchGraph(
        4,
        [
            edge(0, 1, 0.9),
            edge(1, 3, 0.9),
            edge(0, 2, 0.1),
            edge(2, 3, 0.1),
        ],
    )

    plan = graph.plan_homographies(reference_tile=0)

    assert plan.failed_tiles == []
    assert plan.selected_paths[3] == [3, 1, 0]


def test_match_graph_reports_disconnected_tiles():
    graph = MatchGraph(3, [edge(0, 1, 0.9)])

    plan = graph.plan_homographies(reference_tile=0)

    assert plan.connected_tiles == [0, 1]
    assert plan.failed_tiles == [2]
    assert plan.warnings == ["Disconnected tiles: [2]"]
