try:
    import graphviz
    _ = graphviz.Graph().pipe()
    from .diagrams_graphviz import GraphMachine, Graph, NestedGraph, TransitionGraphSupport
except (ImportError, graphviz.ExecutableNotFound):
    from .diagrams_pygraphviz import GraphMachine, Graph, NestedGraph, TransitionGraphSupport
