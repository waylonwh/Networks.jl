# Networks.jl

Network and graph analysis toolkit allowing to handle multigraphs and self-loops.

This package was developed when I was taking a course on Complex Network Modelling and Inference at The University of Adelaide in 2025.
Further development and maintenance is not guaranteed.
Contributions are welcome.

## Installation

First, clone the repository:
```bash
git clone git@github.com:waylonwh/Networks.jl.git
```

Open Julia from within the local directory of the repo via:
```bash
julia --project
```

The first time, you need to install any dependencies:
```julia
julia> using Pkg; Pkg.instantiate()
```

## API reference (AI generated)
**Networks.jl — Public API Reference**

- **Module:** `Networks` (file: `src/Networks.jl`)
- **Purpose:** Lightweight toolkit for graph/network modelling and analysis. Supports multigraphs and self-loops.

**Exported Symbols**
- **`Node`**: Node type and constructors.
- **`Edge`**: Edge type and constructors.
- **`Graph`**: Graph container type and constructors.
- **`add_node!`**: Mutating helpers to add nodes.
- **`add_edge!`**: Mutating helpers to add edges.
- **`node_degrees`**: Compute node degrees.
- **`neighbour_list`**: Adjacency-list representation.
- **`adjacency_matrix`**: Adjacency-matrix representation.
- **`cluster_coeff`**: Global clustering coefficient.

**Submodule**
- **`Networks.Structure`** — internal helpers for structure exploration. Export: **`strong_ccs`** (find strongly connected components).

**Types & Constructors**
- **`Node{K,A}`** — holds an `id::K` and attributes `attr::A` (where `A<:Dict`).
  - `Node{K,A}(id::K, attr::A=A()) where {K,A<:Dict}` — typed constructor.
  - `Node(id::K, attr::Union{Dict,Nothing}=nothing) where {K}` — convenience constructor; `attr` omitted -> `Dict{Any,Any}`.

- **`Edge{K,A,D}`** — holds `id::K`, `ends::NTuple{2,Node{K,A}}`, `endsid::Tuple{K,K}`, `directed::Bool`, and `attr::A`.
  - `Edge{K,A,D}(id::K, ends::NTuple{2,Node{K,A}}, attr::A=A()) where {K,A,D}`
  - `Edge(id::K, ends::NTuple{2,Node{K,A}}, directed::Bool, attr::A=A()) where {K,A}`
  - `Edge(id::K, nodes::Dict{K,Node{K,A}}, endsid::Tuple{K,K}, directed::Bool, attr::A=A()) where {K,A}`

- **`Graph{K,A,D}`** — holds `id::K`, `nodes::Dict{K,Node{K,A}}`, `edges::Dict{K,Edge{K,A,D}}`, `directed::Bool`, and `attr::A`.
  - `Graph{K,A,D}(id::K, nodes::Dict{K,Node{K,A}}=Dict(), edges::Dict{K,Edge{K,A,D}}=Dict(), attr::A=A())`
  - `Graph(id::K, nodes::Dict{...}, edges::Dict{...}, attr::A=A())` — convenience (infers `A` and `D`).
  - `Graph(id::K, edges::Dict{...}, attr::A=A())` — build from edges only.
  - `Graph(id::K, directed::Bool, attr::Union{Dict,Nothing}=nothing) where {K}` — empty graph with chosen directedness.

**Top-level Functions (behaviour & signatures)**

- **`add_node!`**
  - `add_node!(graph::Graph{K,A,D}, node::Node{K,A})::Node{K,A}` — insert or replace node in `graph.nodes`.
  - `add_node!(graph::Graph{K,A,D}, id::K)::Node{K,A}` — convenience: create node with empty `Dict` attributes.

- **`add_edge!`**
  - `add_edge!(graph::Graph{K,A,D}, edge::Edge{K,A,D})::Edge{K,A,D}` — add edge; missing end nodes are added.
  - `add_edge!(graph::Graph{K,A,D}, id::K, endsid::Tuple{K,K}, attr::A=A())::Edge{K,A,D}` — add by ids (uses `graph.nodes`).

- **`node_degrees`**
  - `node_degrees(graph::Graph{K,A,true})::Dict{K,@NamedTuple{out::Int, in::Int}}` — directed graphs: returns (out,in) per node.
  - `node_degrees(graph::Graph{K,A,false})::Dict{K,Int}` — undirected graphs: returns degree per node.

- **`neighbour_list`**
  - `neighbour_list(graph::Graph{K,A,true})::Dict{K,Vector{K}}` — directed neighbor lists (successors). Multiedges preserved as repeated entries.
  - `neighbour_list(graph::Graph{K,A,false})::Dict{K,Vector{K}}` — undirected neighbor lists; self-loops handled (not double-counted).

- **`adjacency_matrix`**
  - `adjacency_matrix(graph::Graph{K,A,D})::NamedTuple{nodesid::Vector{K}, adjmat::Matrix{Int}}` — returns node order and integer adjacency matrix (counts multiplicity); undirected version symmetrises and averages.

- **`cluster_coeff`**
  - `cluster_coeff(adjmat::Matrix{Int}, directed::Bool; selfloop::Bool=false, multi::Bool=false)::Float64` — compute global clustering coefficient from adjacency matrix. Options:
    - `selfloop` (include self-loops)
    - `multi` (treat multiple edges as distinct)
  - `cluster_coeff(graph::Graph{K,A,D}; selfloop::Bool=false, multi::Bool=false)::Float64` — convenience: computes adjacency matrix and calls above.

**Structure submodule**

- **`strong_ccs(neighbours::Dict{K,Vector{K}})::Set{Set{K}}`** — find strongly connected components from neighbours list.
- **`strong_ccs(graph::Graph{K,A,D})::Set{Set{K}}`** — overload: computes neighbour list then delegates.

**Notes & Behavioural Details**

- Multigraphs and self-loops: supported across types; adjacency counts reflect multiplicity unless `multi=false` is passed to `cluster_coeff`.
- Many convenience constructors default attribute type to `Dict{Any,Any}` when attributes are omitted.
- Type parameter `D` on `Edge{...,D}` and `Graph{...,D}` is a Bool value indicating directedness; convenience functions accept `directed::Bool` and construct typed instances.
- `Networks.Structure` contains helper types (`Tree`, `TreeNode`, `CrossTo`, `BackTo`, ...) used internally by `strong_ccs` but not exported at the top-level.

**Examples (short)**

- Create a simple directed graph:
  ```
  nodes = Node.(1:3)
  edges = Dict(-k => Edge(-k, (nodes[k], nodes[k+1]), true) for k in 1:2)
  g = Graph(0, nodes, edges)
  ```

- Compute degrees and clustering:
  ```
  node_degrees(g)
  cluster_coeff(g)
  adjacency_matrix(g)
  ```

**File**: `src/Networks.jl` (primary implementation)
