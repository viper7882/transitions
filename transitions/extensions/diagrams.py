try:
    import graphviz
    try:
        _ = graphviz.Graph().pipe()
    except graphviz.ExecutableNotFound:
        raise ImportError
    from .diagrams_graphviz import GraphMachine, Graph, NestedGraph, TransitionGraphSupport
except ImportError:
    from .diagrams_pygraphviz import GraphMachine, Graph, NestedGraph, TransitionGraphSupport
