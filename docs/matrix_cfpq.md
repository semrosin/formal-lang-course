# Matrix-based context-free path queries

`project.matrix_cfpq.matrix_based_cfpq` implements the matrix algorithm from
[Azimov and Grigorev, *Context-Free Path Querying by Matrix Multiplication*](https://arxiv.org/pdf/1707.01007).
It returns vertex pairs connected by a path whose labels form a word in the
input grammar's language.

## Algorithm

1. Convert the grammar to weak Chomsky normal form using task 6's
   `cfg_to_weak_normal_form`: rules have the form `A -> a`, `A -> B C`, or
   `A -> epsilon`.
2. Convert the graph with `graph_to_nfa` and `AdjacencyMatrixFA` from earlier
   tasks. This provides a compact index for every state and a sparse Boolean
   adjacency matrix for each edge label.
3. Allocate a Boolean CSR matrix `M_A` for every nonterminal. For each rule
   `A -> a`, add the graph's matrix for label `a` to `M_A`. For `A -> epsilon`,
   add the identity matrix, allowing paths of length zero at isolated vertices
   as well.
4. Repeatedly apply `M_A = M_A + M_B @ M_C` for every rule `A -> B C` until
   no matrix gains entries. Boolean addition is union; Boolean multiplication
   finds paths through a shared intermediate vertex. Each pass processes all
   binary rules, including those whose inputs have not changed.
5. Return entries of the grammar's start-symbol matrix that match the requested
   start and final vertices, translating matrix indices back to vertex values.

Entries only change from false to true. There are at most `|N| * |V|^2` true
entries across the nonterminal matrices, so the iteration terminates. Since
the matrices contain no explicit zeros, unchanged `nnz` counts detect the fixed
point. The matrices can become dense even for a sparse input graph.

## API and example

Both `networkx.DiGraph` and `networkx.MultiDiGraph` are supported. Edges carry
their labels in the `label` attribute. Neither the grammar nor the graph is
modified. As in earlier tasks, `None` or an empty set selects all graph vertices;
explicit filter vertices absent from the graph are treated as isolated states.
Filters only restrict the output, so paths may pass through any graph vertex.

```python
import networkx as nx
from pyformlang.cfg import CFG

from project.matrix_cfpq import matrix_based_cfpq

graph = nx.MultiDiGraph()
graph.add_edge(10, 20, label="a")
graph.add_edge(20, 30, label="b")
grammar = CFG.from_text("S -> a S b | a b | $")

assert matrix_based_cfpq(grammar, graph, {10}, {30}) == {(10, 30)}
```

Run the focused tests with:

```shell
python -m pytest tests/test_matrix_cfpq.py tests/autotests/test_task07.py
```

Run the full suite with `python scripts/run_tests.py`.
