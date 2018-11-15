"""
    transitions.extensions.diagrams
    -------------------------------

    Graphviz support for (nested) machines. This also includes partial views
    of currently valid transitions.
"""

import logging
from functools import partial
from collections import defaultdict
from os.path import splitext

from ..core import Transition
from .markup import MarkupMachine
from .nesting import NestedState
try:
    import graphviz as pgv
except ImportError:  # pragma: no cover
    pgv = None

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())

# this is a workaround for dill issues when partials and super is used in conjunction
# without it, Python 3.0 - 3.3 will not support pickling
# https://github.com/pytransitions/transitions/issues/236
_super = super


class Graph(object):
    """ Graph creation for transitions.core.Machine.
        Attributes:
            machine_attributes (dict): Parameters for the general layout of the graph (flow direction, strict etc.)
            style_attributes (dict): Contains style parameters for nodes, edges and the graph
            machine (object): Reference to the related machine.
    """

    machine_attributes = {
        'directed': 'true',
        'strict': 'false',
        'rankdir': 'LR'
    }

    style_attributes = {
        'node': {
            '': {},
            'default': {
                'shape': 'rectangle',
                'style': 'rounded, filled',
                'fillcolor': 'white',
                'color': 'black',
                'peripheries': '1'
            },
            'active': {
                'color': 'red',
                'fillcolor': 'darksalmon',
                'peripheries': '2'
            },
            'previous': {
                'color': 'blue',
                'fillcolor': 'azure2',
                'peripheries': '1'
            }
        },
        'edge': {
            '': {},
            'default': {
                'color': 'black'
            },
            'previous': {
                'color': 'blue'
            }
        },
        'graph': {
            '': {},
            'default': {
                'color': 'black',
                'fillcolor': 'white'
            },
            'previous': {
                'color': 'blue',
                'fillcolor': 'azure2',
                'style': 'filled'
            },
            'active': {
                'color': 'red',
                'fillcolor': 'darksalmon',
                'style': 'filled'
            },
        }
    }

    def __init__(self, machine):
        self.machine = machine
        self.markup = self.machine.markup
        self.roi_state = None
        self.custom_styles = None
        self.reset_styling()

    def set_previous_transition(self, src, dst):
        self.custom_styles['edge'][src][dst] = 'previous'
        self.custom_styles['node'][src] = 'previous'
        self.set_active_state(dst)

    def set_active_state(self, state):
        self.custom_styles['node'][state] = 'active'

    def reset_styling(self):
        self.custom_styles = {'edge': defaultdict(lambda: defaultdict(str)),
                              'node': defaultdict(str)}

    def _add_nodes(self, states, container):
        for state in states:
            style = self.custom_styles['node'][state['name']]
            container.node(state['name'], label=self._convert_state_attributes(state),
                           **self.style_attributes['node'][style])

    def _add_edges(self, transitions, container):
        edge_labels = defaultdict(lambda: defaultdict(list))
        for transition in transitions:
            try:
                dst = transition['dest']
            except KeyError:
                dst = transition['source']
            edge_labels[transition['source']][dst].append(self._transition_label(transition))
        for src, dests in edge_labels.items():
            for dst, labels in dests.items():
                style = self.custom_styles['edge'][src][dst]
                container.edge(src, dst, label=' | '.join(labels), **self.style_attributes['edge'][style])

    def _transition_label(self, tran):
        edge_label = tran.get('label', tran['trigger'])
        if 'dest' not in tran:
            edge_label += " [internal]"
        if self.machine.show_conditions and any(prop in tran for prop in ['conditions', 'unless']):
            x = '{edge_label} [{conditions}]'.format(
                edge_label=edge_label,
                conditions=' & '.join(tran.get('conditions', []) + ['!' + u for u in tran.get('unless', [])]),
            )
            return x
        return edge_label

    def generate(self, title=None, roi_state=None):
        """ Generate a DOT graph with graphviz
        Args:
            title (string): Optional title for the graph.
            roi_state (string): Optional, show only custom states and edges from roi_state
        """
        if not pgv:  # pragma: no cover
            raise Exception('AGraph diagram requires graphviz')

        if title is False:
            title = ''

        fsm_graph = pgv.Digraph(name=title, node_attr=self.style_attributes['node']['default'],
                                edge_attr=self.style_attributes['edge']['default'],
                                graph_attr=self.style_attributes['graph']['default'])
        fsm_graph.graph_attr.update(**self.machine_attributes)
        # For each state, draw a circle
        try:
            states = self.markup.get('states', [])
            transitions = self.markup.get('transitions', [])
            roi_state = self.roi_state if roi_state is None else roi_state
            if roi_state:
                transitions = [t for t in transitions
                               if t['source'] == roi_state or self.custom_styles['edge'][t['source']][t['dest']]]
                state_names = [t for trans in transitions
                               for t in [trans['source'], trans.get('dest', trans['source'])]]
                state_names += [k for k, style in self.custom_styles['node'].items() if style]
                states = _filter_states(states, state_names)
            self._add_nodes(states, fsm_graph)
            self._add_edges(transitions, fsm_graph)
        except KeyError:
            _LOGGER.error("Graph creation incomplete!")
        return fsm_graph

    def draw(self, filename, format=None, prog='dot', args=''):
        """ Generates and saves an image of the state machine using graphviz.
        Args:
            filename (string): path and name of image output
            format (string): Optional format of the output file
        Returns:

        """
        filename, ext = splitext(filename)
        print(filename, ext)
        format = format if format is not None else ext[1:]
        graph = self.generate()
        graph.engine = prog
        graph.render(filename, format=format if format else 'png', cleanup=True)

    def _convert_state_attributes(self, state):
        label = state.get('label', state['name'])
        if self.machine.show_state_attributes:
            if 'tags' in state:
                label += ' [' + ', '.join(state['tags']) + ']'
            if 'on_enter' in state:
                label += '\l- enter:\l  + ' + '\l  + '.join(state['on_enter'])
            if 'on_exit' in state:
                label += '\l- exit:\l  + ' + '\l  + '.join(state['on_exit'])
            if 'timeout' in state:
                label += '\l- timeout(' + state['timeout'] + 's)  -> (' + ', '.join(state['on_timeout']) + ')'
        return label


class NestedGraph(Graph):
    """ Graph creation support for transitions.extensions.nested.HierarchicalGraphMachine.
    Attributes:
        machine_attributes (dict): Same as Graph but extended with cluster/subgraph information
    """

    machine_attributes = Graph.machine_attributes.copy()
    machine_attributes.update(
        {'rank': 'source', 'rankdir': 'TB', 'compound': 'true'})

    def __init__(self, *args, **kwargs):
        _super(NestedGraph, self).__init__(*args, **kwargs)
        self._cluster_states = []

    def _add_nodes(self, states, container, prefix=''):

        for state in states:
            name = prefix + state['name']
            label = self._convert_state_attributes(state)

            if 'children' in state:
                cluster_name = "cluster_" + name
                with container.subgraph(name=cluster_name,
                                        graph_attr=self.style_attributes['graph']['default']) as sub:
                    style = self.custom_styles['node'][name]
                    sub.graph_attr.update(label=label, rank='source', **self.style_attributes['graph'][style])
                    self._cluster_states.append(name)
                    with sub.subgraph(name=cluster_name + '_root',
                                      graph_attr={'label': '', 'color': 'None', 'rank': 'min'}) as root:
                        root.node(name + "_anchor", shape='point', fillcolor='black', width='0.1')
                    self._add_nodes(state['children'], sub, prefix=prefix + state['name'] + NestedState.separator)
            else:
                style = self.custom_styles['node'][name]
                container.node(name, label=label, **self.style_attributes['node'][style])

    def _add_edges(self, transitions, container):
        edges_attr = defaultdict(lambda: defaultdict(dict))

        for transition in transitions:
            # enable customizable labels
            label_pos = 'label'
            src = transition['source']
            try:
                dst = transition['dest']
            except KeyError:
                dst = src
            if edges_attr[src][dst]:
                attr = edges_attr[src][dst]
                attr[attr['label_pos']] = ' | '.join([edges_attr[src][dst][attr['label_pos']],
                                                      self._transition_label(transition)])
                continue
            else:
                attr = {}
                if src in self._cluster_states:
                    attr['ltail'] = 'cluster_' + src
                    src_name = src + "_anchor"
                    label_pos = 'headlabel'
                else:
                    src_name = src

                if dst in self._cluster_states:
                    if not src.startswith(dst):
                        attr['lhead'] = "cluster_" + dst
                        label_pos = 'taillabel' if label_pos.startswith('l') else 'label'
                    dst_name = dst + '_anchor'
                else:
                    dst_name = dst

                # remove ltail when dst (ltail always starts with 'cluster_') is a child of src
                if 'ltail' in attr and dst_name.startswith(attr['ltail'][8:]):
                    del attr['ltail']

                # # remove ltail when dst is a child of src
                # if 'ltail' in edge_attr:
                #     if _get_subgraph(container, edge_attr['ltail']).has_node(dst_name):
                #         del edge_attr['ltail']

                attr[label_pos] = self._transition_label(transition)
                attr['label_pos'] = label_pos
                attr['source'] = src_name
                attr['dest'] = dst_name
                edges_attr[src][dst] = attr

        for src, dests in edges_attr.items():
            for dst, attr in dests.items():
                del attr['label_pos']
                style = self.custom_styles['edge'][src][dst]
                attr.update(**self.style_attributes['edge'][style])
                container.edge(attr.pop('source'), attr.pop('dest'), **attr)


class TransitionGraphSupport(Transition):
    """ Transition used in conjunction with (Nested)Graphs to update graphs whenever a transition is
        conducted.
    """

    def _change_state(self, event_data):
        model = event_data.model
        graph = model.get_graph()
        graph.reset_styling()
        graph.set_previous_transition(self.source, self.dest)
        _super(TransitionGraphSupport, self)._change_state(event_data)  # pylint: disable=protected-access


class GraphMachine(MarkupMachine):
    """ Extends transitions.core.Machine with graph support.
        Is also used as a mixin for HierarchicalMachine.
        Attributes:
            _pickle_blacklist (list): Objects that should not/do not need to be pickled.
            transition_cls (cls): TransitionGraphSupport
    """

    _pickle_blacklist = ['model_graphs']
    graph_cls = Graph
    transition_cls = TransitionGraphSupport

    # model_graphs cannot be pickled. Omit them.
    def __getstate__(self):
        # self.pkl_graphs = [(g.markup, g.custom_styles) for g in self.model_graphs]
        return {k: v for k, v in self.__dict__.items() if k not in self._pickle_blacklist}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model_graphs = {}  # reinitialize new model_graphs

    def __init__(self, *args, **kwargs):
        # remove graph config from keywords
        self.title = kwargs.pop('title', 'State Machine')
        self.show_conditions = kwargs.pop('show_conditions', False)
        self.show_state_attributes = kwargs.pop('show_state_attributes', False)
        # in MarkupMachine this switch is called 'with_auto_transitions'
        # keep 'show_auto_transitions' for backwards compatibility
        kwargs['with_auto_transitions'] = kwargs.pop('show_auto_transitions', False)
        self.model_graphs = {}
        _super(GraphMachine, self).__init__(*args, **kwargs)

        # Create graph at beginning
        for model in self.models:
            if hasattr(model, 'get_graph'):
                raise AttributeError('Model already has a get_graph attribute. Graph retrieval cannot be bound.')
            setattr(model, 'get_graph', partial(self._get_graph, model))
            _ = model.get_graph()  # initialises graph
        # for backwards compatibility assign get_combined_graph to get_graph
        # if model is not the machine
        if not hasattr(self, 'get_graph'):
            setattr(self, 'get_graph', self.get_combined_graph)

    def _get_graph(self, model, title=None, force_new=False, show_roi=False):
        if force_new:
            self.model_graphs[model] = self.graph_cls(self)
            self.model_graphs[model].set_active_state(model.state)
        try:
            m = self.model_graphs[model]
        except KeyError:
            m = self._get_graph(model, title, force_new=True)
        m.roi_state = model.state if show_roi else None
        return m

    def get_combined_graph(self, title=None, force_new=False, show_roi=False):
        """ This method is currently equivalent to 'get_graph' of the first machine's model.
        In future releases of transitions, this function will return a combined graph with active states
        of all models.
        Args:
            title (str): Title of the resulting graph.
            force_new (bool): If set to True, (re-)generate the model's graph.
            show_roi (bool): If set to True, only render states that are active and/or can be reached from
                the current state.
        Returns: AGraph of the first machine's model.
        """
        _LOGGER.info('Returning graph of the first model. In future releases, this '
                     'method will return a combined graph of all models.')
        return self._get_graph(self.models[0], title, force_new, show_roi)

    def set_edge_state(self, graph, edge_from, edge_to, state='default', label=None):
        """ Retrieves/creates an edge between two states and changes the style/label.
        Args:
            graph (AGraph): The graph to be changed.
            edge_from (str): Source state of the edge.
            edge_to (str): Destination state of the edge.
            state (str): Style name (Should be part of the node style_attributes in Graph)
            label (str): Label of the edge.
        """
        # If show_auto_transitions is True, there will be an edge from 'edge_from' to 'edge_to'.
        # This test is considered faster than always calling 'has_edge'.
        if not self.with_auto_transitions and not graph.has_edge(edge_from, edge_to):
            graph.add_edge(edge_from, edge_to, label)
        edge = graph.get_edge(edge_from, edge_to)
        self.set_edge_style(graph, edge, state)

    def add_states(self, states, on_enter=None, on_exit=None,
                   ignore_invalid_triggers=None, **kwargs):
        """ Calls the base method and regenerates all models's graphs. """
        _super(GraphMachine, self).add_states(states, on_enter=on_enter, on_exit=on_exit,
                                              ignore_invalid_triggers=ignore_invalid_triggers, **kwargs)
        for model in self.models:
            model.get_graph(force_new=True)

    def add_transition(self, trigger, source, dest, conditions=None,
                       unless=None, before=None, after=None, prepare=None, **kwargs):
        """ Calls the base method and regenerates all models's graphs. """
        _super(GraphMachine, self).add_transition(trigger, source, dest, conditions=conditions, unless=unless,
                                                  before=before, after=after, prepare=prepare, **kwargs)
        for model in self.models:
            model.get_graph(force_new=True)


def _filter_states(states, state_names, prefix=[]):
    result = []
    for state in states:
        pref = prefix + [state['name']]
        if 'children' in state:
            state['children'] = _filter_states(state['children'], state_names, prefix=pref)
            result.append(state)
        elif NestedState.separator.join(pref) in state_names:
            result.append(state)
    return result
