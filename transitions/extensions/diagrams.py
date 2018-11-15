import warnings

from .. import use_pygraphviz

# make deprecation warnings of transition visible for module users
warnings.filterwarnings(action='default', message=r".*transitions version.*")

try:
    if use_pygraphviz:
        raise ImportError
    import graphviz
    try:
        _ = graphviz.Graph().pipe()
    except graphviz.ExecutableNotFound:
        raise ImportError
    from .diagrams_graphviz import GraphMachine, Graph, NestedGraph, TransitionGraphSupport
except ImportError:
    warnings.warn("Starting from transitions version 0.7.0 graphviz will be the default for creating graphs. "
                  "Fallback to pygraphviz since graphviz could not be found or 'dot' is not in your PATH. "
                  "Please consider installing graphviz.", DeprecationWarning)
    from .diagrams_pygraphviz import GraphMachine, Graph, NestedGraph, TransitionGraphSupport
