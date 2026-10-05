"""Direct unit tests for the multiple-source RPQ solver."""

from networkx import MultiDiGraph

from project.msbfs_rpq import ms_bfs_based_rpq


def test_ms_bfs_based_rpq_matches_expected_pairs() -> None:
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 3, label="b")
    graph.add_edge(0, 2, label="a")
    graph.add_edge(2, 3, label="b")
    graph.add_edge(4, 5, label="a")
    graph.add_edge(5, 7, label="b")

    assert ms_bfs_based_rpq("a b", graph, {0, 4}, {3, 7}) == {
        (0, 3),
        (4, 7),
    }


def test_ms_bfs_based_rpq_uses_all_nodes_when_start_and_final_are_empty() -> None:
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")

    assert ms_bfs_based_rpq("a", graph, set(), set()) == {(0, 1)}


def test_ms_bfs_based_rpq_treats_absent_vertices_as_isolated_states() -> None:
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")

    assert ms_bfs_based_rpq("a b", graph, {0, 99}, {2, 42}) == {(0, 2)}
    assert ms_bfs_based_rpq("a*", graph, {99}, {99}) == {(99, 99)}
