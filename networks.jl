#=
Julia code for
    Complex Network Modelling and Inference: Assignment 6
in Semester 1, 2025.
Module `Networks`. The infrastructure for the script for the assignment.

To use the module:
    julia> include("networks.jl"); using .Networks

External packages required: EzXML
To install the external packages:
    julia> import Pkg; Pkg.add("EzXML")

last updated on 26 Apr 2025.
Waylon Wu
=#


"""
    Networks

Network and graph analysis toolkit allowing to handle multigraphs and self-loops.
"""
module Networks

export
    Node,
    Edge,
    Graph,
    add_node!,
    add_edge!,
    node_degrees,
    neighbour_list,
    adjacency_matrix,
    cluster_coeff

"""
    Node{K,A}

A node with an ID of type `K` and an attribute dictionary of type `A`.
```
"""
struct Node{K,A}
    "Node ID"
    id::K
    "Node attributes"
    attr::A
    @doc """
        Node{K,A<:Dict}(id::K, attr::A<:Dict=A())

    Create a node with an ID of type `K` and an attribute dictionary of type `A`. The type
    of `A` must be a subtype of `Dict`. If `attr` is not provided, an empty attribute
    dictionary of type `A` is created.

    # Examples
    ```jldoctest
    julia> Node{Int,Dict{Symbol,Float64}}(1, Dict(:x => 0.4, :y =>0.5))
    Node{Int64, Dict{Symbol, Float64}}(1) with 2 attributes:
    :y, :x
    ```
    """
    Node{K,A}(id::K, attr::A=A()) where {K,A<:Dict} = new{K,A}(id, attr)
end

"""
    Node(id::K, attr::Union{Dict,Nothing}=nothing)

Create a node. The type of `attr` must be a subtype of `Dict`. If `attr` is not provided, or
is `nothing`, an empty attribute dictionary of type `Dict{Any,Any}` is created.

# Examples
```jldoctest
julia> Node(1, Dict(:x => 0.4, :y =>0.5))
Node{Int64, Dict{Symbol, Float64}}(1) with 2 attributes:
  :y, :x

julia> Node(1)
Node{Int64, Dict{Any, Any}}(1)
```
"""
function Node(id::K, attr::Union{Dict,Nothing}=nothing) where {K}
    if isnothing(attr)
        attr = Dict{Any,Any}()
    end
    return Node{K,typeof(attr)}(id, attr)
end

# Node(id)
(Base.show(io::IO, node::Node{K,A})::Nothing) where {K,A} = print(
    io, "Node(", repr(node.id), ")"
)

#=
Node{K, A}(id)

Node{K, A}(id) with m attribute(s):
  _, _, _, _, _, _, _, _, _, _, ...
=#
function Base.show(io::IO, ::MIME"text/plain", node::Node{K,A})::Nothing where {K,A}
    if length(get(io, :SHOWN_SET, Node{K,A}[])) > 0
        show(io, node) # as a part of container
    elseif isempty(node.attr)
        print(io, "Node{", K, ", ", A, "}(", repr(node.id), ")")
    else
        println(
            io,
            "Node{", K, ", ", A, "}(", repr(node.id), ") with ", length(node.attr),
            "attribute", (length(node.attr) > 1 ? "s" : ""), ":"
        )
        print(io, "  ", join(repr.(first(keys(node.attr), 10)), ", "))
        if length(node.attr) > 10
            print(io, ", ...")
        end
    end
    return nothing
end

"""
    Edge{K,A,D}

Edge with an ID of type `K`, an attribute dictionary of type `A`, and a directed boolean
value `D`. The type of `A` must be a subtype of `Dict`.
"""
struct Edge{K,A,D}
    "Edge ID"
    id::K
    "Source and target nodes"
    ends::NTuple{2,Node{K,A}}
    "Source and target node IDs"
    endsid::Tuple{K,K}
    "Directed edge flag"
    directed::Bool
    "Edge attributes"
    attr::A
    @doc """
        Edge{K,A,D}(id::K, ends::NTuple{2,Node{K,A}}, attr::A=A())

    Create an edge connecting from the source node `ends[1]` to the target node `ends[2]`.
    The type of `attr` must be a subtype of `Dict`. If `attr` is not provided, an empty
    attribute dictionary of type `A` is created. The type of `D` must be a value of type
    `Bool`. If `D` is `true`, the edge is directed from the source node to the target node.
    If `D` is `false`, the edge is undirected.

    # Examples
    ```jldoctest
    julia> Edge{Int,Dict{Any,Any},true}(-1, (Node(1), Node(2)), Dict{Any,Any}(:weight => 2))
    Directed Edge{Int64, Dict{Any, Any}, true}(-1) connecting:
      Node(1) -> Node(2)
    with 1 attribute:
      :weight

    julia> Edge{Int,Dict{Any,Any},false}(-1, (Node(1), Node(2)))
    Undirected Edge{Int64, Dict{Any, Any}, false}(-1) connecting:
      Node(1) -- Node(2)
    ```
    """
    function Edge{K,A,D}(id::K, ends::NTuple{2,Node{K,A}}, attr::A=A()) where {K,A,D}
        if length(ends) != 2
            throw(ArgumentError("Edge must have exactly two ends."))
        elseif !isa(D, Bool)
            throw(ArgumentError("D must be a value of type Bool."))
        end
        endsid = Tuple{K,K}(map((node::Node{K,A} -> node.id), ends))
        return new{K,A,D}(id, ends, endsid, D, attr)
    end
end

"""
    Edge(id::K, ends::NTuple{2,Node{K,A}}, directed::Bool, attr::A=A())

Create an edge. If `directed` is `true`, the edge is directed, otherwise it is undirected.
If `attr` is not provided, or is `nothing`, an empty attribute dictionary of type `A` is
created.

# Examples
```jldoctest
julia> Edge(-1, (Node(1), Node(2)), true, Dict{Any,Any}(:weight => 3))
Directed Edge{Int64, Dict{Any, Any}, true}(-1) connecting:
  Node(1) -> Node(2)
with 1 attribute:
  :weight

julia> Edge(-1, (Node(1), Node(2)), false)
Undirected Edge{Int64, Dict{Any, Any}, false}(-1) connecting:
  Node(1) -- Node(2)
```
"""
Edge(
    id::K, ends::NTuple{2,Node{K,A}}, directed::Bool, attr::A=A()
) where {K,A} = Edge{K,A,directed}(id, ends, attr)

"""
    Edge(id::K, nodes::Dict{K,Node{K,A}}, endsid::Tuple{K,K}, directed::Bool, attr::A=A())

Create an edge connecting the nodes with the given IDs. `nodes` is a dictionary of nodes
with their IDs as keys. If any IDs in `endsid` does not exist in `nodes`, an error is
thrown. If `attr` is not provided, or is `nothing`, an empty attribute dictionary of type A
is created.

# Examples
```jldoctest
julia> Edge(-1, Dict(1:2 .=> Node.(1:2)), (1, 2), true, Dict{Any,Any}(:weight => 3))
Directed Edge{Int64, Dict{Any, Any}, true}(-1) connecting:
  Node(1) -> Node(2)
with 1 attribute:
  :weight

julia> Edge(-1, Dict(1:2 .=> Node.(1:2)), (1, 2), false)
Undirected Edge{Int64, Dict{Any, Any}, false}(-1) connecting:
  Node(1) -- Node(2)
```
"""
function Edge(
    id::K, nodes::Dict{K,Node{K,A}}, endsid::Tuple{K,K}, directed::Bool, attr::A=A()
) where {K,A}
    if !haskey(nodes, endsid[1]) || !haskey(nodes, endsid[2])
        throw(ArgumentError("Nodes with the given IDs don't exist."))
    end
    ends::NTuple{2,Node{K,A}} = (nodes[endsid[1]], nodes[endsid[2]])
    return Edge{K,A,directed}(id, ends, attr)
end

# Edge(eid, nid1->nid2)
# Edge(eid, nid1--nid2)
(Base.show(io::IO, edge::Edge{K,A,D})::Nothing) where {K,A,D} = print(
    io,
    "Edge(", repr(edge.id), ", ", repr(edge.endsid[1]), (D ? "->" : "--"),
    repr(edge.endsid[2]), ")"
)

#=
Directed Edge{K, A, true}(eid) connecting:
  Node(nid1) -> Node(nid2)
with m attribute(s):
  _, _, _, _, _, _, _, _, _, _, ...

Undirected Edge{K, A, false}(eid) connecting:
  Node(nid1) -- Node(nid2)
with m attribute(s):
  _, _, _, _, _, _, _, _, _, _, ...

Directed Edge{K, A, true}(eid) connecting:
  Node(nid1) -> Node(nid2)

Undirected Edge{K, A, false}(eid) connecting:
  Node(nid1) -- Node(nid2)
=#
function Base.show(io::IO, ::MIME"text/plain", edge::Edge{K,A,D})::Nothing where {K,A,D}
    println(
        io,
        (D ? "Directed" : "Undirected"), " Edge{", K, ", ", A, ", ", D, "}(", repr(edge.id),
        ") connecting:"
    )
    print(
        io,
        "  Node(", repr(edge.endsid[1]), ") ", (D ? "->" : "--"), " Node(",
        repr(edge.endsid[2]), ")"
    )
    if !isempty(edge.attr)
        println(
            io,
            "\nwith ", length(edge.attr), " attribute", (length(edge.attr) > 1 ? "s" : ""),
            ":"
        )
        print(io, "  ", join(repr.(first(keys(edge.attr), 10)), ", "))
        if length(edge.attr) > 10
            print(io, ", ...")
        end
    end
    return nothing
end

"""
    Graph{K,A,D}

Graph with an ID of type `K`, an attribute dictionary of type `A`, and a directed boolean
value `D`, where `D` is `true` for directed graphs and `false` for undirected graphs. The
type of `A` must be a subtype of `Dict`. Multigraph is allowed.
"""
struct Graph{K,A,D}
    "Graph ID"
    id::K
    "Nodes in the graph"
    nodes::Dict{K,Node{K,A}}
    "Edges in the graph"
    edges::Dict{K,Edge{K,A,D}}
    "Directed graph flag"
    directed::Bool
    "Graph attributes"
    attr::A
    @doc """
        Graph{K,A,D}(
            id::K,
            nodes::Dict{K,Node{K,A}}=Dict{K,Node{K,A}}(),
            edges::Dict{K,Edge{K,A,D}}=Dict{K,Edge{K,A,D}}(),
            attr::A=A()
        )

    Create a graph with nodes and edges. If any ends in the `edges` do not exist in the
    `nodes`, the node would be added. If `nodes`, `edges`, or `attr` are not provided, empty
    dictionaries of type `Dict{K,Node{K,A}}`, `Dict{K,Edge{K,A,D}}`, and `A` are created,
    respectively. The type of `D` must be a value of type `Bool`.

    # Examples
    ```jldoctest
    julia> nodes = Node.(1:2)
    2-element Vector{Node{Int64, Dict{Any, Any}}}:
     Node(1)
     Node(2)

    julia> edges = Dict(
               -1 => Edge(-1, (nodes[1], nodes[2]), true),
               -2 => Edge(-2, (nodes[1], Node(3)), true)
           )
    Dict{Int64, Edge{Int64, Dict{Any, Any}, true}} with 2 entries:
      -1 => Edge(-1, 1->2)
      -2 => Edge(-2, 1->3)

    julia> Graph{Int,Dict{Any,Any},true}(
               0, nodes, edges, Dict{Any,Any}(:name => "example graph")
           )
    Directed Graph{Int64, Dict{Any, Any}, true}(0) containing 3 nodes:
      2, 3, 1
    and 2 edges:
      -1, -2
    with 1 attribute:
      :name

    julia> Graph{Int,Dict{Any,Any},false}(0)
    Undirected Graph{Int64, Dict{Any, Any}, false}(0) is empty.
    ```
    """
    function Graph{K,A,D}(
        id::K,
        nodes::Dict{K,Node{K,A}}=Dict{K,Node{K,A}}(),
        edges::Dict{K,Edge{K,A,D}}=Dict{K,Edge{K,A,D}}(),
        attr::A=A()
    ) where {K,A,D}
        nodes = copy(nodes) # to protect the original dictionary
        # add node if not already present
        for edge::Edge{K,A,D} in values(edges)
            for i::Int in 1:2
                if !haskey(nodes, edge.endsid[i])
                    nodes[edge.endsid[i]] = edge.ends[i]
                end
            end
        end
        return new{K,A,D}(id, nodes, edges, D, attr)
    end
end

"""
    Graph(id::K, nodes::Dict{K,Node{K,A}}, edges::Dict{K,Edge{K,A,D}}, attr::A=A())

Create a graph with nodes and edges. If `attr` is not provided, an empty attribute
dictionary of type `A` is created. The attribute dictionary type `A` and directed boolean
value `D` is inferred from the type of values in `edges` dictionary.

# Examples
```jldoctest
julia> nodes = Node.(1:3);

julia> edges = Dict(-k => Edge(-k, (nodes[k], nodes[k+1]), true) for k in 1:2);

julia> Graph(0, nodes, edges)
Directed Graph{Int64, Dict{Any, Any}, true}(0) containing 3 nodes:
  2, 3, 1
and 2 edges:
  -1, -2
```
"""
Graph(
    id::K, nodes::Dict{K,Node{K,A}}, edges::Dict{K,Edge{K,A,D}}, attr::A=A()
) where {K,A,D} = Graph{K,A,D}(id, nodes, edges, attr)

"""
    Graph(id::K, edges::Dict{K,Edge{K,A,D}}, attr::A=A())

Create a graph spanned by the edges. If `attr` is not provided, an empty attribute
dictionary of type `A` is created.

# Examples
```jldoctest
julia> nodes = Node.(1:4);

julia> edges = Dict(-k => Edge(-k, (nodes[k], nodes[k+1]), true) for k in 1:3);

julia> Graph(0, edges)
Directed Graph{Int64, Dict{Any, Any}, true}(0) containing 4 nodes:
  4, 2, 3, 1
and 3 edges:
  -1, -3, -2
```
"""
Graph(id::K, edges::Dict{K,Edge{K,A,D}}, attr::A=A()) where {K,A,D} = Graph(
    id, Dict{K,Node{K,A}}(), edges, attr
)

"""
    Graph(id::K, directed::Bool, attr::Union{Dict,Nothing}=nothing)

Create an empty graph with the given ID, directedness, and attributes. If `attr` is not
provided, or is `nothing`, an empty attribute dictionary of type `Dict{Any,Any}` is created.

# Examples
```jldoctest
julia> Graph("empty graph", false)
Undirected Graph{String, Dict{Any, Any}, false}("empty graph") is empty.
```
"""
function Graph(id::K, directed::Bool, attr::Union{Dict,Nothing}=nothing) where {K}
    if isnothing(attr)
        attr = Dict{Any,Any}()
    end
    attr_type::DataType = typeof(attr)
    return Graph{K,attr_type,directed}(
        id, Dict{K,Node{K,attr_type}}(), Dict{K,Edge{K,attr_type,directed}}(), attr
    )
end

# Graph(id, directed, |N|=n, |E|=e)
# Graph(id, undirected, |N|=n, |E|=e)
(Base.show(io::IO, graph::Graph{K,A,D})::Nothing) where {K,A,D} = print(
    io,
    "Graph(", repr(graph.id), ", ", (D ? "directed" : "undirected"), ", |N|=",
    length(graph.nodes), ", |E|=", length(graph.edges), ")"
)

#=
Directed Graph{K, A, true}(id) is empty.

Undirected Graph{K, A, false}(id) is empty.

Directed Graph{K, A, true}(id) containing n node(s):
  _, _, _, _, _, _, _, _, _, _, ...
and no edges.

Undirected Graph{K, A, false}(id) containing n node(s):
  _, _, _, _, _, _, _, _, _, _, ...
and no edges.

Directed Graph{K, A, true}(id) containing n node(s):
  _, _, _, _, _, _, _, _, _, _, ...
and e edge(s):
  _, _, _, _, _, _, _, _, _, _, ...

Directed Graph{K, A, true}(0) containing n node(s):
  _, _, _, _, _, _, _, _, _, _, ...
and e edge(s):
  _, _, _, _, _, _, _, _, _, _, ...
with a attribute(s):
  _, _, _, _, _, _, _, _, _, _, ...
=#
function Base.show(io::IO, ::MIME"text/plain", graph::Graph{K,A,D}) where {K,A,D}
    if length(get(io, :SHOWN_SET, Graph{K,A,D}[])) > 0
        show(io, graph) # as a part of container
    else
        print(
            io,
            (D ? "Directed" : "Undirected"),
            " Graph{", K, ", ", A, ", ", D, "}(", repr(graph.id), ") "
        )
        if isempty(graph.nodes)
            print(io, "is empty.")
        else
            println(
                io,
                "containing ", length(graph.nodes), " node",
                (length(graph.nodes) > 1 ? "s" : ""), ":"
            )
            print(io, "  ", join(repr.(first(keys(graph.nodes), 10)), ", "))
            if length(graph.nodes) > 10
                print(io, ", ...")
            end
            if isempty(graph.edges)
                print(io, "\nand no edges", (isempty(graph.attr) ? '.' : ','))
            else
                println(
                    io,
                    "\nand ", length(graph.edges), " edge",
                    (length(graph.edges) > 1 ? "s" : ""), ":"
                )
                print(io, "  ", join(repr.(first(keys(graph.edges), 10)), ", "))
                if length(graph.edges) > 10
                    print(io, ", ...")
                end
            end
            if !isempty(graph.attr)
                println(
                    io,
                    "\nwith ", length(graph.attr), " attribute",
                    (length(graph.attr) > 1 ? "s" : ""), ":"
                )
                print(io, "  ", join(repr.(first(keys(graph.attr), 10)), ", "))
                if length(graph.attr) > 10
                    print(io, ", ...")
                end
            end
        end
    end
    return nothing
end

"""
    add_node!(graph::Graph{K,A,D}, node::Node{K,A}) -> Node{K,A}

Add the `node` to the `graph`. If the node already exists, it is replaced with the new node.

# Examples
```jldoctest
julia> graph = Graph(0, true);

julia> add_node!(Graph(0, true), Node(1))
Node{Int64, Dict{Any, Any}}(1)

julia> graph
Directed Graph{Int64, Dict{Any, Any}, true}(0) containing 1 node:
  1
and no edges.
```
"""
(add_node!(graph::Graph{K,A,D}, node::Node{K,A})::Node{K,A}) where {K,A,D} = (
    graph.nodes[node.id] = node
)

"""
    add_node!(graph::Graph{K,A,D}, id::K) -> Node{K,A}

Add the node with the given ID and empty attribute dictionary of type `A` to the `graph`.

# Examples
```jldoctest
julia> graph = Graph(0, false);

julia> add_node!(graph, 1)
Node{Int64, Dict{Any, Any}}(1)

julia> graph
Undirected Graph{Int64, Dict{Any, Any}, false}(0) containing 1 node:
  1
and no edges.
```
"""
(add_node!(graph::Graph{K,A,D}, id::K)::Node{K,A}) where {K,A,D} = add_node!(
    graph, Node{K,A}(id, A())
)

"""
    add_edge!(graph::Graph{K,A,D}, edge::Edge{K,A,D}) -> Edge{K,A,D}

Add the `edge` to the `graph`. If any ends in the `edge` don't exist in the `graph`, the
node would be also added. If the edge already exists, it is replaced with the new edge.

# Examples
```jldoctest
julia> node1 = Node(1);

julia> graph = Graph{Int,Dict{Any,Any},true}(0, Dict(1 => node1))
Directed Graph{Int64, Dict{Any, Any}, true}(0) containing 1 node:
  1
and no edges.

julia> add_edge!(graph, Edge(-1, (node1, Node(2)), true))
Directed Edge{Int64, Dict{Any, Any}, true}(-1) connecting:
  Node(1) -> Node(2)

julia> graph
Directed Graph{Int64, Dict{Any, Any}, true}(0) containing 2 nodes:
  2, 1
and 1 edge:
  -1
```
"""
function add_edge!(graph::Graph{K,A,D}, edge::Edge{K,A,D})::Edge{K,A,D} where {K,A,D}
    # add node if not already present
    for node::Node{K,A} in edge.ends
        if !haskey(graph.nodes, node.id)
            add_node!(graph, node)
        end
    end
    # add edge
    graph.edges[edge.id] = edge
    return edge
end

"""
    add_edge!(graph::Graph{K,A,D}, id::K, endsid::Tuple{K,K}, attr::A=A()) -> Edge{K,A,D}

Add the edge with the given ID that connects the nodes with the given IDs to the `graph`. If
any nodes with their IDs do not exist in the `graph`, an error is thrown. If the attribute
dictionary of the edge `attr` is not provided, an empty attribute dictionary of type `A` is
created.

# Examples
```jldoctest
julia> graph = Graph{Int,Dict{Any,Any},true}(0, Dict(1:2 .=> Node.(1:2)))
Directed Graph{Int64, Dict{Any, Any}, true}(0) containing 2 nodes:
  2, 1
and no edges.

julia> add_edge!(graph, -2, (2, 1))
Directed Edge{Int64, Dict{Any, Any}, true}(-2) connecting:
  Node(2) -> Node(1)

julia> graph
Directed Graph{Int64, Dict{Any, Any}, true}(0) containing 2 nodes:
  2, 1
and 1 edge:
  -2
```
"""
(
    add_edge!(graph::Graph{K,A,D}, id::K, endsid::Tuple{K,K}, attr::A=A())::Edge{K,A,D}
) where {K,A,D} = add_edge!(graph, Edge(id, graph.nodes, endsid, D, attr))

"""
    Networks.asdirected(edge::Edge{K,A,false}) -> Edge{K,A,true}

Convert an undirected edge to a directed edge, remaining other fields of the `edge`
unchanged. The direction is set from the first node to the second node in `edge.ends`.
```
"""
(asdirected(edge::Edge{K,A,false})::Edge{K,A,true}) where {K,A} = Edge{K,A,true}(
    edge.id, edge.ends, edge.attr
)

"""
    Networks.asdirected(graph::Graph{K,A,false}) -> Graph{K,A,true}

Convert an undirected graph to a directed graph, remaining other fields of the `graph`
unchanged. Each undirected edge in the graph is converted to a directed edge, with the
direction set from the first node to the second node in the `ends` field of the edge.
"""
function asdirected(graph::Graph{K,A,false})::Graph{K,A,true} where {K,A}
    directed_edges = Dict{K,Edge{K,A,true}}(
        map(
            (edge::Edge{K,A,false} -> (edge.id => asdirected(edge))),
            collect(values(graph.edges))
        )
    )
    pseudodirected = Graph{K,A,true}(graph.id, graph.nodes, directed_edges, graph.attr)
    return pseudodirected
end

"""
    node_degrees(graph::Graph{K,A,true}) -> Dict{K,@NamedTuple{out::Int, in::Int}}

Calculate the out and in degrees of each node in the directed `graph`. A dictionary with the
node IDs as keys and a named tuple with the `:out` and `:in` degrees as values is returned.

# Examples
```jldoctest
julia> nodes = Node.(1:3);

julia> edges = Dict(-k => Edge(-k, (nodes[k], nodes[k+1]), true) for k in 1:2)
Dict{Int64, Edge{Int64, Dict{Any, Any}, true}} with 2 entries:
  -1 => Edge(-1, 1->2)
  -2 => Edge(-2, 2->3)

julia> node_degrees(Graph(0, edges))
Dict{Int64, @NamedTuple{out::Int64, in::Int64}} with 3 entries:
  2 => (out = 1, in = 1)
  3 => (out = 0, in = 1)
  1 => (out = 1, in = 0)
```
"""
function node_degrees(
    graph::Graph{K,A,true}
)::Dict{K,@NamedTuple{out::Int, in::Int}} where {K,A}
    mutdegs = Dict{K,Vector{Int}}(
        keys(graph.nodes) .=> ([0, 0] for _ in eachindex(graph.nodes))
    )
    for edge::Edge{K,A,true} in values(graph.edges)
        mutdegs[edge.endsid[1]][1] += 1 # out degree
        mutdegs[edge.endsid[2]][2] += 1 # in degree
    end
    degrees = Dict{K,@NamedTuple{out::Int, in::Int}}(
        map(
            (id::K -> (id => NamedTuple{(:out, :in)}(mutdegs[id]))),
            collect(keys(graph.nodes))
        )
    )
    return degrees
end

"""
    node_degrees(graph::Graph{K,A,false}) -> Dict{K,Int}

Calculate the degree of each node in the undirected `graph`. A dictionary with the node IDs
as keys and the degree as values is returned.

# Examples
```jldoctest
julia> nodes = Node.(1:3);

julia> edges = Dict(-k => Edge(-k, (nodes[k], nodes[k+1]), false) for k in 1:2)
Dict{Int64, Edge{Int64, Dict{Any, Any}, false}} with 2 entries:
  -1 => Edge(-1, 1--2)
  -2 => Edge(-2, 2--3)

julia> node_degrees(Graph(0, edges))
Dict{Int64, Int64} with 3 entries:
  2 => 2
  3 => 1
  1 => 1
```
"""
function node_degrees(graph::Graph{K,A,false})::Dict{K,Int} where {K,A}
    oidegs::Dict{K,@NamedTuple{out::Int, in::Int}} = node_degrees(asdirected(graph))
    degrees = Dict{K,Int}(
        map((id::K -> (id => sum(oidegs[id]))), collect(keys(graph.nodes)))
    )
    return degrees
end

"""
    neighbour_list(graph::Graph{K,A,true}) -> Dict{K,Vector{K}}

Get the neighbour list representation of the directed `graph`. A dictionary with the node
IDs as keys and a vector of neighbour node IDs as values is returned.

# Examples
```jldoctest
julia> nodes = Node.(1:3);

julia> edges = Dict(
           [
               (-k => Edge(-k, (nodes[k], nodes[k+1]), true) for k in 1:2)...,
               -3 => Edge(-3, (nodes[3], nodes[3]), true),
               -4 => Edge(-4, (nodes[1], nodes[2]), true)
           ]
       )
Dict{Int64, Edge{Int64, Dict{Any, Any}, true}} with 4 entries:
  -1 => Edge(-1, 1->2)
  -3 => Edge(-3, 3->3)
  -2 => Edge(-2, 2->3)
  -4 => Edge(-4, 1->2)

julia> neighbour_list(Graph(0, edges))
Dict{Int64, Vector{Int64}} with 3 entries:
  2 => [3]
  3 => [3]
  1 => [2, 2]
```
"""
function neighbour_list(graph::Graph{K,A,true})::Dict{K,Vector{K}} where {K,A}
    neighbours = Dict{K,Vector{K}}(
        collect(keys(graph.nodes)) .=> fill(Vector{K}[], length(graph.nodes))
    )
    foreach(
        (edge::Edge{K,A,true} -> push!(neighbours[edge.endsid[1]], edge.endsid[2])),
        values(graph.edges)
    )
    return neighbours
end

"""
    neighbour_list(graph::Graph{K,A,false}) -> Dict{K,Vector{K}}

Get the neighbour list representation of the undirected `graph`.

# Examples
```jldoctest
julia> nodes = Node.(1:3);

julia> edges = Dict(
           [
               (-k => Edge(-k, (nodes[k], nodes[k+1]), false) for k in 1:2)...,
               -3 => Edge(-3, (nodes[3], nodes[3]), false),
               -4 => Edge(-4, (nodes[1], nodes[2]), false)
           ]
       )
Dict{Int64, Edge{Int64, Dict{Any, Any}, false}} with 4 entries:
  -1 => Edge(-1, 1--2)
  -3 => Edge(-3, 3--3)
  -2 => Edge(-2, 2--3)
  -4 => Edge(-4, 1--2)

julia> neighbour_list(Graph(0, edges))
Dict{Int64, Vector{Int64}} with 3 entries:
  2 => [3, 1, 1]
  3 => [3, 2]
  1 => [2, 2]
```
"""
function neighbour_list(graph::Graph{K,A,false})::Dict{K,Vector{K}} where {K,A}
    neighbours::Dict{K,Vector{K}} = neighbour_list(asdirected(graph))
    for edge::Edge{K,A,false} in values(graph.edges)
        if edge.endsid[1] != edge.endsid[2] # avoid double counting self loops
            push!(neighbours[edge.endsid[2]], edge.endsid[1])
        end
    end
    return neighbours
end

"""
    adjacency_matrix(
        graph::Graph{K,A,D}
    ) -> @NamedTuple{nodesid::Vector{K}, adjmat::Matrix{Int}}

Get the adjacency matrix representation of the directed `graph`. A named tuple with fields
`nodesid` and `adjmat` is returned. The `nodesid` field contains a vector of node IDs, and
the `adjmat` field contains the adjacency matrix where the index of the row and column
corresponds to the node IDs in `nodesid`.

# Examples
```jldoctest
julia> nodes = Node.(1:3);

julia> edges = Dict(
           [
               (-k => Edge(-k, (nodes[k], nodes[k+1]), true) for k in 1:2)...,
               -3 => Edge(-3, (nodes[3], nodes[3]), true),
               -4 => Edge(-4, (nodes[1], nodes[2]), true)
           ]
       )
Dict{Int64, Edge{Int64, Dict{Any, Any}, true}} with 4 entries:
  -1 => Edge(-1, 1->2)
  -3 => Edge(-3, 3->3)
  -2 => Edge(-2, 2->3)
  -4 => Edge(-4, 1->2)


julia> key_mat = adjacency_matrix(Graph(0, edges))
(nodesid = [2, 3, 1], adjmat = [0 1 0; 0 1 0; 2 0 0])

julia> key_mat.nodesid
3-element Vector{Int64}:
 2
 3
 1

julia> key_mat.adjmat
3×3 Matrix{Int64}:
 0  1  0
 0  1  0
 2  0  0
```
"""
function adjacency_matrix(
    graph::Graph{K,A,true}
)::@NamedTuple{nodesid::Vector{K}, adjmat::Matrix{Int}} where {K,A}
    nodesid::Vector{K} = collect(keys(graph.nodes))
    id2ix = Dict{K,Int}(nodesid .=> eachindex(nodesid))
    adjmat::Matrix{Int} = zeros(Int, length(nodesid), length(nodesid))
    for edge::Edge{K,A,true} in values(graph.edges)
        adjmat[id2ix[edge.endsid[1]], id2ix[edge.endsid[2]]] += 1
    end
    key_mat::@NamedTuple{nodesid::Vector{K}, adjmat::Matrix{Int}} = (; nodesid, adjmat)
    return key_mat
end

"""
    adjacency_matrix(
        graph::Graph{K,A,false}
    ) -> @NamedTuple{nodesid::Vector{K}, adjmat::Matrix{Int}}

Get the adjacency matrix representation of the undirected `graph`.

# Examples
```jldoctest
julia> nodes = Node.(1:3);

julia> edges = Dict(
           [
               (-k => Edge(-k, (nodes[k], nodes[k+1]), false) for k in 1:2)...,
               -3 => Edge(-3, (nodes[3], nodes[3]), false),
               -4 => Edge(-4, (nodes[1], nodes[2]), false)
           ]
       )
Dict{Int64, Edge{Int64, Dict{Any, Any}, false}} with 4 entries:
  -1 => Edge(-1, 1--2)
  -3 => Edge(-3, 3--3)
  -2 => Edge(-2, 2--3)
  -4 => Edge(-4, 1--2)

julia> key_mat = adjacency_matrix(Graph(0, edges));

julia> key_mat.nodesid
3-element Vector{Int64}:
 2
 3
 1

julia> key_mat.adjmat
3×3 Matrix{Int64}:
 0  1  2
 1  1  0
 2  0  0
```
"""
function adjacency_matrix(
    graph::Graph{K,A,false}
)::@NamedTuple{nodesid::Vector{K}, adjmat::Matrix{Int}} where {K,A}
    half_key_mat::@NamedTuple{nodesid::Vector{K}, adjmat::Matrix{Int}} = adjacency_matrix(
        asdirected(graph)
    )
    adjmat::Matrix{Int} = half_key_mat.adjmat + half_key_mat.adjmat'
    adjmat[axes(adjmat, 1).==axes(adjmat, 1)'] ./= 2 # remove self loops
    key_mat = NamedTuple{(:nodesid, :adjmat)}(
        (half_key_mat.nodesid, adjmat)
    )
    return key_mat
end

"""
    Networks.triplet_via(n::Int, adjmat::Matrix{Int}, directed::Bool) -> Int

Calculate the number of routes of length 2 (triplets) via the node `n` in the graph
represented by the adjacency matrix `adjmat`.
"""
function triplet_via(n::Int, adjmat::Matrix{Int}, directed::Bool)::Int
    if directed
        triplet::Int = sum(adjmat[:, n]) * sum(adjmat[n, :]) - adjmat[n, n]
    else # !directed
        degree::Int = sum(adjmat[n, :])
        triplet = degree * (degree - 1)
    end
    return triplet
end

"""
    Networks.triangle_via(p::Int, n::Int, s::Int, adjmat::Matrix{Int}) -> Int

Calculate the number of closed routes of length of 3 (triangles) sequentially passing
through `p`, `n`, and `s` in the directed graph represented by the adjacency matrix
`adjmat`. Three nodes are allowed to be non-unique.
"""
function triangle_via(p::Int, n::Int, s::Int, adjmat::Matrix{Int})::Int
    if p == n == s # loop three times via itself
        triangle::Int = maximum(
            (0, (adjmat[p, n] * (adjmat[n, s] - 1) * (adjmat[s, p] - 2)))
        )
        #   if number of self loop < 3, no triangles form (0 triangles)
    else # three paths are unique
        @inbounds triangle = adjmat[p, n] * adjmat[n, s] * adjmat[s, p]
    end
    return triangle
end

"""
    Networks.triangle_via(
        p::Int, n::Int, s::Int, adjmat::Matrix{Int}, directed::Bool
    ) -> Int

Calculate the number of closed routes of length of 3 (triangles) sequentially passing
through `p`, `n`, and `s` in the graph represented by the adjacency matrix. Boolean
`directed` tells the directeness of the graph.
"""
function triangle_via(p::Int, n::Int, s::Int, adjmat::Matrix{Int}, directed::Bool)::Int
    if directed || (p == n == s || p != n != s != p)
    #   3 loops or no loop in an undirected graph
        triangle::Int = triangle_via(p, n, s, adjmat)
    else # 1 self loop
        nodes::Vector{Int} = unique((p, n, s))
        n2i = Dict{Int,Int}(nodes .=> 1:2)
        minimat::Matrix{Int} = adjmat[nodes, nodes]
        triangle = 1
        for (s::Int, t::Int) in ((p, n), (n, s), (s, p))
            triangle *= maximum((0, minimat[n2i[s], n2i[t]]))
            minimat[n2i[s], n2i[t]] -= 1
            minimat[n2i[t], n2i[s]] -= 1 # retain the symmetry
        end
    end
    return triangle
end

"""
    cluster_coeff(
        adjmat::Matrix{Int}, directed::Bool; selfloop::Bool=false, multi::Bool=false
    ) -> Float64

Calculate the global clustering coefficient of a graph represented by the adjacency matrix.
If the graph is directed, the adjacency matrix must be symmetric. The `selfloop` argument
specifies whether to consider self loops in the calculation. Selfloops are ignored by
default. The `multi` argument specifies whether to consider multiple edges between nodes
as different edges. Multiple edges are considered as one edge by default.

# Examples
```jldoctest
julia> di_adjmat = [
               1 1 1 2 1;
               0 0 1 0 0;
               1 0 0 1 0;
               0 0 0 0 1;
               1 1 0 0 0;
           ];

julia> cluster_coeff(di_adjmat, true)
0.3

julia> cluster_coeff(di_adjmat, true, selfloop=true)
0.46153846153846156

julia> cluster_coeff(di_adjmat, true, multi=true)
0.391304347826087

julia> cluster_coeff(di_adjmat, true, selfloop=true, multi=true)
0.5

julia> un_adjmat = di_adjmat + di_adjmat';

julia> un_adjmat[axes(un_adjmat, 1).==axes(un_adjmat, 1)'] ./= 2 #=undirected graph=#;

julia> cluster_coeff(un_adjmat, false)
0.6666666666666666

julia> cluster_coeff(un_adjmat, false, selfloop=true)
0.5454545454545454

julia> cluster_coeff(un_adjmat, false, multi=true)
0.8571428571428571

julia> cluster_coeff(un_adjmat, false, selfloop=true, multi=true)
0.9183673469387755
```
"""
function cluster_coeff(
    adjmat::Matrix{Int}, directed::Bool; selfloop::Bool=false, multi::Bool=false
)::Float64
    if !directed && adjmat != adjmat' # check input arguments
        throw(
            ArgumentError("Adjacency matrix must be symmetric for undirected graphs.")
        )
    end
    if !selfloop
        adjmat = copy(adjmat) # protect the original matrix
        adjmat[axes(adjmat, 1).==axes(adjmat, 1)'] .= 0 # remove self loops
    end
    if !multi
        adjmat = Matrix{Int}(adjmat .> 0) # remove multiple edges
    end
    triplet::Int = 0
    triangle::Int = 0
    for n::Int in axes(adjmat, 1)
        tris_vian::Int = triplet_via(n, adjmat, directed)
        triplet += tris_vian
        if tris_vian > 0 # looking for possible triangles
            pres::Vector{Int} = findall(adjmat[:, n] .!= 0) # predecessors
            sucs::Vector{Int} = findall(adjmat[n, :] .!= 0) # successors
            for p::Int in pres
                for s::Int in sucs
                    triangle += triangle_via(p, n, s, adjmat, directed)
                end
            end
        end
    end
    coefficient::Float64 = triangle / triplet
    return coefficient
end

"""
    cluster_coeff(graph::Graph{K,A,D}; selfloop::Bool=false, multi::Bool=false) -> Float64

Calculate the global clustering coefficient of a given `graph`.

# Examples
```jldoctest
julia> nodes = Dict(0:4 .=> Node.(0:4));

julia> edge_list = ((0, 1), (0, 3), (0, 4), (1, 2), (2, 0), (3, 4));

julia> di_edges = Dict([-e[1] => Edge(e[1], nodes, e[2], true) for e in pairs(edge_list)]);

julia> cluster_coeff(Graph(10, nodes, di_edges))
0.5

julia> ud_edges = Dict([-e[1] => Edge(e[1], nodes, e[2], false) for e in pairs(edge_list)]);

julia> cluster_coeff(Graph(20, nodes, ud_edges))
0.6
```
"""
function cluster_coeff(
    graph::Graph{K,A,D}; selfloop::Bool=false, multi::Bool=false
)::Float64 where {K,A,D}
    adjmat::Matrix{Int} = adjacency_matrix(graph).adjmat
    coefficient::Float64 = cluster_coeff(adjmat, D, selfloop=selfloop, multi=multi)
    return coefficient
end


"""
    Networks.Structure

Explore the structure of a network. Optimised for multigraphs.
"""
module Structure

export strong_ccs

using ..Networks

"""
    Structure.AbstractTo{K}

Abstract type for the links in a spanning tree.
"""
abstract type AbstractTo{K} end

"""
    Structure.TreeNode{K} <: AbstractTo{K}
    Structure.TreeNode{K}(id::K, root::K, path::Vector{K})

A node in a spanning tree.
"""
struct TreeNode{K}
    "Node ID"
    id::K
    "The root of the tree that the node belongs to"
    root::K
    "The path to the node from the root"
    path::Vector{K}
end

"""
    Structure.BranchTo{K} <: AbstractTo{K}
    Structure.BranchTo{K}(from::TreeNode{K}, to::TreeNode{K})

A link in a spanning tree.
"""
struct BranchTo{K} <: AbstractTo{K}
    "Source of the edge"
    from::TreeNode{K}
    "Target of the edge"
    to::TreeNode{K}
end

"""
    Structure.CrossTo{K} <: AbstractTo{K}
    Structure.CrossTo{K}(from::TreeNode{K}, to::TreeNode{K})

A link cross to another branch or another tree in a spanning tree.
"""
struct CrossTo{K} <: AbstractTo{K}
    "Source of the edge"
    from::TreeNode{K}
    "Target of the edge"
    to::TreeNode{K}
end

"""
    Structure.BackTo{K} <: AbstractTo{K}
    Structure.BackTo{K}(from::TreeNode{K}, to::TreeNode{K})

A link back to a parent node in the same branch in a spanning tree.
"""
struct BackTo{K} <: AbstractTo{K}
    "Source of the edge"
    from::TreeNode{K}
    "Target of the edge"
    to::TreeNode{K}
end

"""
    Structure.Tree{K}

Spanning tree of a graph from a root node, where edges that violate the tree structure are
collected in `crosses` and `backs` field as vectors.
"""
struct Tree{K}
    "The root of the tree"
    root::K
    "An index of the nodes to their corresponding `TreeNode`s"
    nodes::Dict{K,TreeNode{K}}
    "The tree structure"
    tree::Dict{TreeNode{K},Vector{AbstractTo{K}}}
    "Links to other branches or trees"
    crosses::Vector{CrossTo{K}}
    "Links back to the parent node in the same branch"
    backs::Vector{BackTo{K}}
    @doc """
        Structure.Tree{K}(root::K) where K

    Create a new spanning tree with the given `root` node, where the `root` node is
    included.
    """
    function Tree{K}(root::K) where {K}
        root_node = TreeNode{K}(root, root, [root])
        return new(
            root,
            Dict{K,TreeNode{K}}(root => root_node),
            Dict{TreeNode{K},Vector{AbstractTo{K}}}(root_node => AbstractTo{K}[]),
            CrossTo{K}[],
            BackTo{K}[]
        )
    end
end

"""
    Structure.find_node(trees::Dict{K,Tree{K}}, id::K) -> Union{TreeNode{K},Nothing}

Find the `TreeNode` with the given ID in the spanning trees of a graph. If the node is not
found, `nothing` is returned.
"""
function find_node(trees::Dict{K,Tree{K}}, id::K)::Union{TreeNode{K},Nothing} where {K}
    for tree::Tree{K} in values(trees)
        if haskey(tree.nodes, id)
            return tree.nodes[id] # node found
        end
    end
    return nothing # node does not exist
end

"""
    Structure.span_tree!(
        trees::Dict{K,Tree{K}}, neighbours::Dict{K,Vector{K}}, root::K, visited::Set{K}
    ) -> Tree{K}

Create a spanning tree from the `neighbours` dictionary, starting from the given `root`
untill no more available un`visited` nodes can be reached. The spanned tree is added to the
`trees` dictionary with the `root` as the key, and the `visited` set is updated.
"""
function span_tree!(
    trees::Dict{K,Tree{K}}, neighbours::Dict{K,Vector{K}}, root::K, visited::Set{K}
)::Tree{K} where {K}
    # initialise
    tree = Tree{K}(root)
    unvisited::Set{K} = setdiff(Set{K}(keys(neighbours)), visited)
    queue::Vector{TreeNode{K}} = [tree.nodes[root]]
    while !isempty(queue)
        node::TreeNode{K} = popfirst!(queue)
        for to::K in neighbours[node.id]
            if to in unvisited # BranchTo
                # add to the tree
                tonode = TreeNode{K}(to, root, K[node.path; to])
                tree.nodes[to] = tonode
                tree.tree[tonode] = Vector{AbstractTo{K}}()
                push!(tree.tree[node], BranchTo{K}(node, tonode))
                # update states
                push!(queue, tonode)
                delete!(unvisited, to)
                push!(visited, to)
            elseif to in node.path # BackTo
                backto = BackTo{K}(node, tree.nodes[to])
                push!(tree.tree[node], backto)
                push!(tree.backs, backto)
            else # CrossTo
                if haskey(tree.nodes, to) # CrossTo another branch
                    tonode = tree.nodes[to]
                    crossto = CrossTo{K}(node, tree.nodes[to])
                    push!(tree.tree[node], crossto)
                else # CrossTo another tree
                    tonode = find_node(trees, to)
                    crossto = CrossTo{K}(node, tonode)
                    push!(tree.tree[node], crossto)
                end
                push!(tree.crosses, crossto)
            end
        end
    end
    trees[root] = tree
    return tree
end

"""
    Structure.astrees(neighbours::Dict{K,Vector{K}}, root::K) -> Dict{K,Tree{K}}

Create a spanning tree with the given `root`. New roots are selected if not all nodes are
reachable from the given `root`, untill all nodes are included in the spanning trees. A
dictionary with the `root` nodes as keys and the corresponding spanning trees as values is
returned.
"""
function astrees(neighbours::Dict{K,Vector{K}}, root::K)::Dict{K,Tree{K}} where {K}
    # initialise
    visited::Set{K} = Set{K}()
    trees::Dict{K,Tree{K}} = Dict{K,Tree{K}}()
    while length(visited) < length(neighbours) # there are other trees to be spanned
        if haskey(trees, root) # restart from another root
            root = first(setdiff(Set{K}(keys(neighbours)), visited))
        end
        push!(visited, root)
        span_tree!(trees, neighbours, root, visited)
    end
    return trees
end

"""
    Structure.merge_nodes!(
        neighbours::Dict{K,Vector{K}}, spanned::Set{K}, nodes::Set{K}, id::K
    ) -> @NamedTuple{id::K, neighbours::Dict{K,Vector{K}}, spanned::Set{K}}

Merge the `nodes` in the `neighbours` dictionary and update the `spanned` set, replacing
them with the new node with `id`. All internal edges among the `nodes` are removed from
the neighbour list. A named tuple with the new node `id`, updated `neighbours`, and updated
`spanned` set is returned. If any elements in `nodes` don't appear in the `neighbours` or
`spanned`, they are ignored.
"""
function merge_nodes!(
    neighbours::Dict{K,Vector{K}},
    spanned::Set{K},
    nodes::Set{K},
    id::K
)::@NamedTuple{id::K, neighbours::Dict{K,Vector{K}}, spanned::Set{K}} where {K}
    # update neighbour lists
    merged::Pair{K,Vector{K}} = (id => K[])
    for node::K in nodes
        if haskey(neighbours, node)
            # find the neighbours of the nodes
            for neighbour::K in neighbours[node]
                if !in(neighbour, nodes)
                    push!(merged[2], neighbour)
                end
            end
            # remove the nodes from the neighbours
            delete!(neighbours, node)
        end
    end
    # replace the nodes appear as neighbours with the new node
    foreach((list::Vector{K} -> replace!(list, (nodes .=> id)...)), values(neighbours))
    # add the new node to the neighbour lists
    push!(neighbours, merged)
    # update spanned set
    replace!(spanned, (nodes .=> id)...)
    return (; id, neighbours, spanned)
end

#=
Update the `hypers` dictionary by merging the loops in the tree structure into it.

If a loop in `loops` can not be identified as the same connected component already existing
in `hypers`, it is added to `hypers` with the node ID representing the connected component
generated by the `idmap` function. The `idmap` function takes an element in `loops`
(a `Set{K}`) as input and returns a valid node ID (with type `K`). The default `idmap`
function is `first`, which returns the first element from iterating the `Set{K}`.

If merging a loop into `hypers` results two connected components in `hypers` can be
identified as the same one, they are merged into one entry in `hypers` with the node ID
representing the connected component is chosen from two of them.

Intended for internal use only.
=#
function merge_loops_into!(
    hypers::Dict{K,Set{K}}, loops::Set{Set{K}}, idmap::Function=first
)::Dict{K,Set{K}} where {K}
    # merge ccs to hypers
    for loop::Set{K} in loops
        unitedto::Union{K,Nothing} = nothing
        for (hyper::K, cluster::Set{K}) in hypers
            if (hyper in loop) || !isempty(intersect(cluster, loop))
            #   cc is a part of another cc
                if isnothing(unitedto) # merge cc to hyper
                    union!(cluster, setdiff(loop, Ref(hyper)))
                    unitedto = hyper
                else # merge hyper to united hyper
                    setdiff!(hypers[unitedto], Ref(hyper))
                    union!(hypers[unitedto], cluster)
                    delete!(hypers, hyper)
                end
            end
        end
        if isnothing(unitedto) # cc is not in any existing cc
            hypers[idmap(loop)] = loop
        end
    end
    return hypers
end

#=
Find all loops in the spanning trees of a graph. A `Set` of all nodes in the loops as a
`Set{K}` is returned.

Intended for internal use only.
=#
function find_loops(trees::Dict{K,Tree{K}})::Set{Set{K}} where {K}
    loops = Set{Set{K}}()
    for tree::Tree{K} in values(trees)
        for back::BackTo{K} in tree.backs
            start_inx::Int = findfirst(isequal(back.to.id), back.from.path)
            loop = Set{K}(back.from.path[start_inx:end])
            push!(loops, loop)
        end
    end
    return loops
end

"""
    Structure.find_hyper(hypers::Dict{K,Set{K}}, id::K) -> K

Find the node ID representing the connected component in which the given node with `id`
belongs.
"""
function find_hyper(hypers::Dict{K,Set{K}}, id::K)::K where {K}
    for (hyper::K, cluster::Set{K}) in hypers
        if id in cluster
            return hyper
        end
    end
    return id # id is not in any hyper (as a hyper as itself)
end

#=
Find a possible root node that spanning trees from it can forms loops in the trees (there
are `CrossTo` edges from it). Nodes that are already `spanned` are excluded. Nodes in
`spanned` are allowed to be the nodes in the original graph or nodes representing a
connected components. If it's the former, the node representing the connected component in
which the node belongs is returned. `nothing` is returned if no such root is found.

Intended for internal use only.
=#
function find_unspanned_root(
    trees::Dict{K,Tree{K}}, spanned::Set{K}, hypers::Dict{K,Set{K}}
)::Union{K,Nothing} where {K}
    for tree::Tree{K} in values(trees)
        cross_inx::Union{Int,Nothing} = findfirst(
            (link::CrossTo{K} -> !in(find_hyper(hypers, link.from.id), spanned)),
            tree.crosses
        )
        if !isnothing(cross_inx) # found a root that is not in spanned
            root::K = find_hyper(hypers, tree.crosses[cross_inx].from.id)
            return root
        end
    end
    return nothing # no unspanned root
end

"""
    strong_ccs(neighbours::Dict{K,Vector{K}}) -> Set{Set{K}}

Find all strongly connected components in the graph represented by the `neighbours` list.

# Examples
```jldoctest
julia> neighbours = Dict(1=>[2], 2=>[3,4], 3=>[4,6,6], 4=>[1,5], 5=>[5,6], 6=>[7], 7=>[5]);

julia> strong_ccs(neighbours)
Set{Set{Int64}} with 2 elements:
  Set([4, 2, 3, 1])
  Set([5, 6, 7])
```
"""
function strong_ccs(neighbours::Dict{K,Vector{K}})::Set{Set{K}} where {K}
    neighbours = copy(neighbours) # to protect the original dictionary
    hyper_nodes = Dict{K,Set{K}}()
    spanned_roots = Set{K}()
    root::Union{K,Nothing} = first(keys(neighbours))
    while !isnothing(root)
        trees::Dict{K,Tree{K}} = astrees(neighbours, root)
        push!(spanned_roots, root)
        loops::Set{Set{K}} = find_loops(trees)
        merge_loops_into!(hyper_nodes, loops, first)
        for (id::K, cluster::Set{K}) in hyper_nodes
            merge_nodes!(neighbours, spanned_roots, cluster, id)
        end
        root = find_unspanned_root(trees, spanned_roots, hyper_nodes)
    end
    # add individual nodes
    ccs = Set{Set{K}}(values(hyper_nodes))
    union!(ccs, Set{K}.(Ref.(setdiff(keys(neighbours), keys(hyper_nodes)))))
    return ccs
end

"""
    strong_ccs(graph::Graph{K,A,D}) -> Set{Set{K}}

Find all strongly connected components in the directed `graph{K,A,true}`, or connected
components in the undirected `graph{K,A,false}`.

# Examples
```jldoctest
julia> nodes = Dict(1:5 .=> Node.(1:5));

julia> edgelist = ((1,3), (3,2), (2,1), (1,4), (4,5));

julia> diedges = Dict(-k => Edge(-k,nodes,edgelist[k],true) for k in eachindex(edgelist));

julia> strong_ccs(Graph(10, nodes, diedges))
Set{Set{Int64}} with 3 elements:
  Set([5])
  Set([2, 3, 1])
  Set([4])

julia> udedges = Dict(-k => Edge(-k,nodes,edgelist[k],false) for k in eachindex(edgelist));

julia> strong_ccs(Graph(20, nodes, udedges))
Set{Set{Int64}} with 1 element:
  Set([5, 4, 2, 3, 1])
```
"""
function strong_ccs(graph::Graph{K,A,D})::Set{Set{K}} where {K,A,D}
    neighbours::Dict{K,Vector{K}} = neighbour_list(graph)
    ccs::Set{Set{K}} = strong_ccs(neighbours)
    return ccs
end

end # module Structure

end # module Networks
