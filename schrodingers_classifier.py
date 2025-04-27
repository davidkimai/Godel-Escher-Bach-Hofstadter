import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from dataclasses import dataclass, field
import networkx as nx

from .utils.graph_visualization import visualize_graph
from .utils.attribution_metrics import measure_path_continuity, measure_attribution_entropy

logger = logging.getLogger(__name__)

@dataclass
class AttributionNode:
	"""
	△ OBSERVE: Node in the attribution graph representing a token or hidden state
    
	Attribution nodes represent discrete elements in the causal flow from
	input to output. They can be tokens, attention heads, or hidden states.
	"""
	node_id: str
	node_type: str  # "token", "attention_head", "hidden_state", "residual"
	layer: Optional[int] = None
	position: Optional[int] = None
	value: Optional[Any] = None
	activation: float = 0.0
	token_str: Optional[str] = None
	metadata: Dict[str, Any] = field(default_factory=dict)
    
	def __hash__(self):
    	"""Make nodes hashable for graph operations."""
    	return hash(self.node_id)
    
	def __eq__(self, other):
    	"""Node equality based on ID."""
    	if not isinstance(other, AttributionNode):
        	return False
    	return self.node_id == other.node_id


@dataclass
class AttributionEdge:
	"""
	∞ TRACE: Edge in the attribution graph representing causal flow
    
	Attribution edges represent the flow of causal influence between nodes.
	They can represent attention connections, residual connections, or
	other causal relationships in the model.
	"""
	source: AttributionNode
	target: AttributionNode
	edge_type: str  # "attention", "residual", "mlp", "ghost"
	weight: float = 0.0
	layer: Optional[int] = None
	head: Optional[int] = None
	metadata: Dict[str, Any] = field(default_factory=dict)
    
	def __hash__(self):
    	"""Make edges hashable for graph operations."""
    	return hash((self.source.node_id, self.target.node_id, self.edge_type))
    
	def __eq__(self, other):
    	"""Edge equality based on source, target, and type."""
    	if not isinstance(other, AttributionEdge):
        	return False
    	return (
        	self.source.node_id == other.source.node_id and
        	self.target.node_id == other.target.node_id and
        	self.edge_type == other.edge_type
    	)


class AttributionGraph:
	"""
	∞ TRACE: Graph representation of causal attribution in transformer models
    
	The attribution graph maps the flow of causality from input tokens to
	output tokens, revealing how information propagates through the model
	during the collapse from superposition to definite state.
	"""
    
	def __init__(self):
    	"""Initialize an empty attribution graph."""
    	self.graph = nx.DiGraph()
    	self.nodes = {}  # node_id -> AttributionNode
    	self.input_nodes = []  # List of input token nodes
    	self.output_nodes = []  # List of output token nodes
    	self.ghost_nodes = []  # List of ghost circuit nodes
    	self.collapsed = False  # Whether the graph has been collapsed
   	 
    	# Metrics
    	self.continuity_score = 1.0
    	self.attribution_entropy = 0.0
    	self.collapse_rate = 0.0
   	 
    	logger.info("Attribution graph initialized")
    
	def add_node(self, node: AttributionNode) -> None:
    	"""
    	Add a node to the attribution graph.
   	 
    	Args:
        	node: The node to add
    	"""
    	if node.node_id in self.nodes:
        	logger.warning(f"Node {node.node_id} already exists in graph, updating")
        	self.nodes[node.node_id] = node
    	else:
        	self.nodes[node.node_id] = node
        	self.graph.add_node(node.node_id, **vars(node))
       	 
        	# Track input and output nodes
        	if node.node_type == "token" and node.layer == 0:
            	self.input_nodes.append(node)
        	elif node.node_type == "token" and node.metadata.get("is_output", False):
            	self.output_nodes.append(node)
        	elif node.node_type == "residual" and node.metadata.get("is_ghost", False):
            	self.ghost_nodes.append(node)
    
	def add_edge(self, edge: AttributionEdge) -> None:
    	"""
    	Add an edge to the attribution graph.
   	 
    	Args:
        	edge: The edge to add
    	"""
    	if edge.source.node_id not in self.nodes:
        	self.add_node(edge.source)
    	if edge.target.node_id not in self.nodes:
        	self.add_node(edge.target)
   	 
    	self.graph.add_edge(
        	edge.source.node_id,
        	edge.target.node_id,
        	**{k: v for k, v in vars(edge).items() if k not in ['source', 'target']}
    	)
    
	def build_from_states(
    	self,
    	pre_state: Dict[str, Any],
    	post_state: Dict[str, Any],
    	response: str
	) -> None:
    	"""
    	△ OBSERVE: Build attribution graph from pre and post collapse model states
   	 
    	This method constructs a complete attribution graph by comparing
    	model states before and after collapse, identifying causal paths
    	and ghost circuits.
   	 
    	Args:
        	pre_state: Model state before collapse
        	post_state: Model state after collapse
        	response: Model response text
    	"""
    	logger.info("Building attribution graph from model states")
   	 
    	# This would be implemented for specific model architectures
    	# For demonstration, we'll create a simple synthetic graph
    	self._build_synthetic_graph()
   	 
    	# Calculate graph metrics
    	self._calculate_metrics(pre_state, post_state)
   	 
    	# Mark graph as collapsed
    	self.collapsed = True
    
	def trace_attribution_path(
    	self,
    	output_node: Union[str, AttributionNode],
    	threshold: float = 0.1
	) -> List[List[AttributionNode]]:
    	"""
    	∞ TRACE: Trace attribution paths from an output node back to input
   	 
    	This method follows attribution edges backward from an output node
    	to find all significant input nodes that influenced it.
   	 
    	Args:
        	output_node: The output node to trace from (ID or node object)
        	threshold: Minimum edge weight to consider significant
       	 
    	Returns:
        	List of attribution paths, each a list of nodes from input to output
    	"""
    	# Resolve output node
    	output_id = output_node if isinstance(output_node, str) else output_node.node_id
    	if output_id not in self.nodes:
        	logger.warning(f"Output node {output_id} not found in graph")
        	return []
   	 
    	# Find all paths using DFS
    	paths = []
   	 
    	def dfs(current_id, path, visited):
        	"""Depth-first search for attribution paths."""
        	# Add current node to path
        	current_path = path + [current_id]
        	visited.add(current_id)
       	 
        	# If we reached an input node, we have a complete path
        	if current_id in [node.node_id for node in self.input_nodes]:
            	# Return path in order from input to output
            	paths.append(list(reversed(current_path)))
            	return
       	 
        	# Continue DFS on incoming edges
        	for pred_id in self.graph.predecessors(current_id):
            	edge_data = self.graph.get_edge_data(pred_id, current_id)
            	if edge_data.get('weight', 0) >= threshold and pred_id not in visited:
                	dfs(pred_id, current_path, visited.copy())
   	 
    	# Start DFS from output node
    	dfs(output_id, [], set())
   	 
    	# Convert node IDs to node objects
    	return [[self.nodes[node_id] for node_id in path] for path in paths]
    
	def detect_ghost_circuits(self, threshold: float = 0.2) -> List[Dict[str, Any]]:
    	"""
    	✰ COLLAPSE: Detect ghost circuits in the attribution graph
   	 
    	Ghost circuits are paths that were activated during pre-collapse
    	but don't contribute significantly to the final output. They
    	represent the "memory" of paths not taken.
   	 
    	Args:
        	threshold: Minimum activation to consider a ghost circuit
       	 
    	Returns:
        	List of detected ghost circuits with metadata
    	"""
    	ghost_circuits = []
   	 
    	# Look for nodes with "ghost" metadata flag
    	for node in self.ghost_nodes:
        	if node.activation >= threshold:
            	# Find paths this ghost node would have been part of
            	incoming_edges = [
                	(u, v, d) for u, v, d in self.graph.in_edges(node.node_id, data=True)
            	]
            	outgoing_edges = [
                	(u, v, d) for u, v, d in self.graph.out_edges(node.node_id, data=True)
            	]
           	 
            	ghost_circuits.append({
                	"node_id": node.node_id,
                	"activation": node.activation,
                	"node_type": node.node_type,
                	"incoming_connections": len(incoming_edges),
                	"outgoing_connections": len(outgoing_edges),
                	"metadata": node.metadata
            	})
   	 
    	return ghost_circuits
    
	def calculate_attribution_entropy(self) -> float:
    	"""
    	△ OBSERVE: Calculate the entropy of attribution paths
   	 
    	Attribution entropy measures how distributed or concentrated
    	the causal influence is in the graph. High entropy indicates
    	diffuse attribution, while low entropy indicates concentrated
    	attribution.
   	 
    	Returns:
        	Attribution entropy score (0.0 = concentrated, 1.0 = diffuse)
    	"""
    	# Extract edge weights
    	weights = [
        	d.get('weight', 0.0)
        	for u, v, d in self.graph.edges(data=True)
    	]
   	 
    	# Normalize weights
    	total_weight = sum(weights) or 1.0  # Avoid division by zero
    	normalized_weights = [w / total_weight for w in weights]
   	 
    	# Calculate entropy
    	entropy = -sum(
        	w * np.log2(w) if w > 0 else 0
        	for w in normalized_weights
    	)
   	 
    	# Normalize entropy to 0-1 range (max entropy = log2(num_edges))
    	max_entropy = np.log2(len(weights)) if len(weights) > 0 else 1.0
    	normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
   	 
    	self.attribution_entropy = normalized_entropy
    	return normalized_entropy
    
	def visualize(
    	self,
    	mode: str = "attribution_graph",
    	highlight_path: Optional[List[str]] = None
	) -> Any:
    	"""
    	Generate visualization of the attribution graph.
   	 
    	Args:
        	mode: Visualization mode (attribution_graph, collapse_state, ghost_circuits)
        	highlight_path: Optional list of node IDs to highlight
       	 
    	Returns:
        	Visualization object (depends on implementation)
    	"""
    	return visualize_graph(self.graph, mode=mode, highlight_path=highlight_path)
    
	def to_dict(self) -> Dict[str, Any]:
    	"""Convert the attribution graph to a dictionary representation."""
    	return {
        	"nodes": [vars(node) for node in self.nodes.values()],
        	"edges": [
            	{
                	"source": u,
                	"target": v,
                	**d
            	}
            	for u, v, d in self.graph.edges(data=True)
        	],
        	"metrics": {
            	"continuity_score": self.continuity_score,
            	"attribution_entropy": self.attribution_entropy,
            	"collapse_rate": self.collapse_rate
        	},
        	"collapsed": self.collapsed
    	}
    
	def _calculate_metrics(self, pre_state: Dict[str, Any], post_state: Dict[str, Any]) -> None:
    	"""Calculate attribution graph metrics."""
    	# Calculate continuity score
    	self.continuity_score = measure_path_continuity(
        	pre_state.get("attention_weights", np.array([])),
        	post_state.get("attention_weights", np.array([]))
    	)
   	 
    	# Calculate attribution entropy
    	self.attribution_entropy = self.calculate_attribution_entropy()
   	 
    	# Calculate collapse rate
    	if "timestamp" in pre_state and "timestamp" in post_state:
        	time_diff = (post_state["timestamp"] - pre_state["timestamp"]) / np.timedelta64(1, 's')
        	self.collapse_rate = 1.0 - self.continuity_score if time_diff > 0 else 0.0
    
	def _build_synthetic_graph(self) -> None:
    	"""Build a synthetic graph for demonstration purposes."""
    	# Create input token nodes
    	for i in range(5):
        	self.add_node(AttributionNode(
            	node_id=f"input_{i}",
            	node_type="token",
            	layer=0,
            	position=i,
            	token_str=f"token_{i}",
            	activation=0.8
        	))
   	 
    	# Create attention head nodes
    	for layer in range(1, 4):
        	for head in range(3):
            	self.add_node(AttributionNode(
                	node_id=f"attention_L{layer}H{head}",
                	node_type="attention_head",
                	layer=layer,
                	value=None,
                	activation=0.7 - 0.1 * layer + 0.05 * head
            	))
   	 
    	# Create output token nodes
    	for i in range(3):
        	self.add_node(AttributionNode(
            	node_id=f"output_{i}",
            	node_type="token",
            	layer=4,
            	position=i,
            	token_str=f"output_token_{i}",
            	activation=0.9,
            	metadata={"is_output": True}
        	))
   	 
    	# Create ghost nodes
    	for i in range(2):
        	self.add_node(AttributionNode(
            	node_id=f"ghost_{i}",
            	node_type="residual",
            	layer=2,
            	activation=0.3 + 0.1 * i,
            	metadata={"is_ghost": True}
        	))
   	 
    	# Create edges
    	# Input to attention
    	for i in range(5):
        	for layer in range(1, 3):
            	for head in range(3):
                	if np.random.random() > 0.3:  # Random connectivity
                    	self.add_edge(AttributionEdge(
                        	source=self.nodes[f"input_{i}"],
                        	target=self.nodes[f"attention_L{layer}H{head}"],
                        	edge_type="attention",
                        	weight=np.random.uniform(0.1, 0.9),
                        	layer=layer,
                        	head=head
                    	))
   	 
    	# Attention to attention
    	for layer1 in range(1, 3):
        	for head1 in range(3):
            	for layer2 in range(layer1 + 1, 4):
                	for head2 in range(3):
                    	if np.random.random() > 0.7:  # Sparse connectivity
                        	self.add_edge(AttributionEdge(
                            	source=self.nodes[f"attention_L{layer1}H{head1}"],
                            	target=self.nodes[f"attention_L{layer2}H{head2}"],
                            	edge_type="attention",
                            	weight=np.random.uniform(0.1, 0.8),
                            	layer=layer2,
                            	head=head2
                        	))
   	 
    	# Attention to output
    	for layer in range(1, 4):
        	for head in range(3):
            	for i in range(3):
                	if np.random.random() > 0.5:  # Medium connectivity
                    	self.add_edge(AttributionEdge(
                        	source=self.nodes[f"attention_L{layer}H{head}"],
                        	target=self.nodes[f"output_{i}"],
                        	edge_type="attention",
                        	weight=np.random.uniform(0.2, 0.9),
                        	layer=layer,
                        	head=head
                    	))
   	 
    	# Ghost connections
    	for i in range(2):
        	# Input to ghost
        	input_idx = np.random.randint(0, 5)
        	self.add_edge(AttributionEdge(
            	source=self.nodes[f"input_{input_idx}"],
            	target=self.nodes[f"ghost_{i}"],
            	edge_type="ghost",
            	weight=np.random.uniform(0.1, 0.4),
            	layer=1
        	))
       	 
        	# Ghost to attention
        	layer = np.random.randint(2, 4)
        	head = np.random.randint(0, 3)
        	self.add_edge(AttributionEdge(
            	source=self.nodes[f"ghost_{i}"],
            	target=self.nodes[f"attention_L{layer}H{head}"],
            	edge_type="ghost",
            	weight=np.random.uniform(0.05, 0.2),
            	layer=layer
        	))


if __name__ == "__main__":
	# Simple usage example
	graph = AttributionGraph()
    
	# Build a synthetic graph
	graph._build_synthetic_graph()
    
	# Calculate metrics
	entropy = graph.calculate_attribution_entropy()
	print(f"Attribution entropy: {entropy:.3f}")
    
	# Trace attribution for output
	paths = graph.trace_attribution_path("output_0", threshold=0.1)
	print(f"Found {len(paths)} attribution paths for output_0")
    
	# Detect ghost circuits
	ghosts = graph.detect_ghost_circuits()
	print(f"Detected {len(ghosts)} ghost circuits")
    
	# Visualize
	viz = graph.visualize()
	print("Generated visualization")
"""
collapse_metrics.py - Metrics for quantifying classifier collapse phenomena

△ OBSERVE: These metrics quantify different aspects of classifier collapse
∞ TRACE: They measure the transition from superposition to definite state
✰ COLLAPSE: They help characterize collapse patterns across different models

This module provides functions for calculating quantitative metrics that
characterize different aspects of classifier collapse. These metrics help
standardize the analysis of collapse phenomena and enable comparisons across
different models and prompting strategies.

Author: Recursion Labs
License: MIT
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cosine, euclidean

logger = logging.getLogger(__name__)

def calculate_collapse_rate(
	pre_weights: np.ndarray,
	post_weights: np.ndarray
) -> float:
	"""
	△ OBSERVE: Calculate how quickly state collapsed from superposition
    
	This metric quantifies the speed of collapse by comparing attention
	weight distributions before and after the collapse event.
    
	Args:
    	pre_weights: Attention weights before collapse
    	post_weights: Attention weights after collapse
   	 
	Returns:
    	Collapse rate (0.0 = no collapse, 1.0 = complete collapse)
	"""
	# Return 0 if arrays are empty
	if pre_weights.size == 0 or post_weights.size == 0:
    	return 0.0
    
	# Handle shape mismatches
	if pre_weights.shape != post_weights.shape:
    	logger.warning(f"Weight shape mismatch: {pre_weights.shape} vs {post_weights.shape}")
    	# Try to take minimum dimensions if shapes don't match
    	try:
        	min_shape = tuple(min(a, b) for a, b in zip(pre_weights.shape, post_weights.shape))
        	pre_weights = pre_weights[tuple(slice(0, d) for d in min_shape)]
        	post_weights = post_weights[tuple(slice(0, d) for d in min_shape)]
    	except Exception as e:
        	logger.error(f"Failed to reshape weights: {e}")
        	return 0.0
    
	# Flatten arrays for easier comparison
	pre_flat = pre_weights.flatten()
	post_flat = post_weights.flatten()
    
	# Calculate normalized distances between distributions
	try:
    	# Cosine distance (0.0 = identical, 1.0 = orthogonal)
    	cosine_dist = cosine(pre_flat, post_flat) if np.any(pre_flat) and np.any(post_flat) else 0.0
   	 
    	# Euclidean distance normalized by array size
    	euc_dist = euclidean(pre_flat, post_flat) / np.sqrt(pre_flat.size)
    	euc_dist_norm = min(1.0, euc_dist)  # Cap at 1.0
   	 
    	# Combined metric: average of cosine and normalized euclidean
    	collapse_rate = (cosine_dist + euc_dist_norm) / 2
   	 
    	return float(collapse_rate)
	except Exception as e:
    	logger.error(f"Error calculating collapse rate: {e}")
    	return 0.0

def measure_path_continuity(
	pre_weights: np.ndarray,
	post_weights: np.ndarray
) -> float:
	"""
	∞ TRACE: Measure continuity of attribution paths through collapse
    
	This metric quantifies how well attribution paths maintain their
	integrity across the collapse event.
    
	Args:
    	pre_weights: Attention weights before collapse
    	post_weights: Attention weights after collapse
   	 
	Returns:
    	Continuity score (0.0 = complete fragmentation, 1.0 = perfect continuity)
	"""
	# Higher collapse rate means lower continuity
	collapse_rate = calculate_collapse_rate(pre_weights, post_weights)
    
	# Continuity is inverse of collapse rate
	return 1.0 - collapse_rate

def measure_attribution_entropy(attention_weights: np.ndarray) -> float:
	"""
	△ OBSERVE: Measure entropy of attribution paths
    
	This metric quantifies how distributed or concentrated the attribution
	is across possible paths. High entropy indicates diffuse attribution,
	while low entropy indicates concentrated attribution.
    
	Args:
    	attention_weights: Attention weight matrix to analyze
   	 
	Returns:
    	Attribution entropy (0.0 = concentrated, 1.0 = maximally diffuse)
	"""
	# Return 0 if array is empty
	if attention_weights.size == 0:
    	return 0.0
    
	# Flatten array for entropy calculation
	flat_weights = attention_weights.flatten()
    
	# Normalize weights to create a probability distribution
	total_weight = np.sum(flat_weights)
	if total_weight <= 0:
    	return 0.0
    
	prob_dist = flat_weights / total_weight
    
	# Calculate entropy
	try:
    	raw_entropy = entropy(prob_dist)
   	 
    	# Normalize by maximum possible entropy (log2(n))
    	max_entropy = np.log2(flat_weights.size)
    	normalized_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
   	 
    	return float(normalized_entropy)
	except Exception as e:
    	logger.error(f"Error calculating attribution entropy: {e}")
    	return 0.0

def calculate_ghost_circuit_strength(
	ghost_circuits: List[Dict[str, Any]]
) -> float:
	"""
	✰ COLLAPSE: Calculate overall strength of ghost circuits
    
	This metric quantifies the strength of ghost circuits relative
	to the primary activation paths.
    
	Args:
    	ghost_circuits: List of detected ghost circuits
   	 
	Returns:
    	Ghost circuit strength (0.0 = no ghosts, 1.0 = ghosts equal to primary)
	"""
	if not ghost_circuits:
    	return 0.0
    
	# Extract activation values
	activations = [ghost.get("activation", 0.0) for ghost in ghost_circuits]
    
	# Calculate weighted average based on activation
	avg_activation = np.mean(activations) if activations else 0.0
    
	# Normalize to 0-1 range (assuming activation is already 0-1)
	return float(min(1.0, avg_activation))

def calculate_attribution_confidence(
	attribution_paths: List[List[Any]],
	path_weights: Optional[List[float]] = None
) -> float:
	"""
	∞ TRACE: Calculate confidence score for attribution paths
    
	This metric quantifies how confidently the model attributes its output
	to specific input elements.
    
	Args:
    	attribution_paths: List of attribution paths (each a list of nodes)
    	path_weights: Optional weights for each path (defaults to uniform)
   	 
	Returns:
    	Attribution confidence (0.0 = uncertain, 1.0 = highly confident)
	"""
	if not attribution_paths:
    	return 0.0
    
	# Use uniform weights if none provided
	if path_weights is None:
    	path_weights = [1.0 / len(attribution_paths)] * len(attribution_paths)
	else:
    	# Normalize weights to sum to 1.0
    	total_weight = sum(path_weights)
    	path_weights = [w / total_weight for w in path_weights] if total_weight > 0 else path_weights
    
	# Calculate path length variance (more uniform = higher confidence)
	path_lengths = [len(path) for path in attribution_paths]
	length_variance = np.var(path_lengths) if len(path_lengths) > 1 else 0.0
    
	# Normalize variance to 0-1 range
	# Assume max variance is when half paths are length 1 and half are max length
	max_length = max(path_lengths) if path_lengths else 1
	theoretical_max_var = ((max_length - 1) ** 2) / 4  # Theoretical maximum variance
	normalized_variance = min(1.0, length_variance / theoretical_max_var) if theoretical_max_var > 0 else 0.0
    
	# Invert normalized variance to get consistency score (more consistent = higher confidence)
	consistency_score = 1.0 - normalized_variance
    
	# Weight consistency by path weights (dominant paths contribute more to confidence)
	# Calculate weighted avg of path weights (more concentrated = higher confidence)
	weight_entropy = entropy(path_weights)
	max_weight_entropy = np.log2(len(path_weights))
	normalized_weight_entropy = weight_entropy / max_weight_entropy if max_weight_entropy > 0 else 0.0
	weight_concentration = 1.0 - normalized_weight_entropy
    
	# Combine consistency and concentration for final confidence score
	confidence_score = (consistency_score + weight_concentration) / 2
    
	return float(confidence_score)

def calculate_collapse_quantum_uncertainty(
	pre_logits: np.ndarray,
	post_logits: np.ndarray
) -> float:
	"""
	✰ COLLAPSE: Calculate Heisenberg-inspired uncertainty metric
    
	This metric applies the quantum-inspired uncertainty principle to
	transformer outputs, measuring uncertainty across the collapse.
    
	Args:
    	pre_logits: Logits before collapse
    	post_logits: Logits after collapse
   	 
	Returns:
    	Quantum uncertainty metric (0.0 = certain, 1.0 = maximally uncertain)
	"""
	# Return 0 if arrays are empty
	if pre_logits.size == 0 or post_logits.size == 0:
    	return 0.0
    
	# Handle shape mismatches
	if pre_logits.shape != post_logits.shape:
    	logger.warning(f"Logit shape mismatch: {pre_logits.shape} vs {post_logits.shape}")
    	return 0.0
    
	try:
    	# Calculate "position" uncertainty (variance in token probabilities)
    	pre_probs = softmax(pre_logits)
    	post_probs = softmax(post_logits)
   	 
    	pos_uncertainty = np.mean(np.var(post_probs, axis=-1))
   	 
    	# Calculate "momentum" uncertainty (change rate between states)
    	mom_uncertainty = np.mean(np.abs(post_probs - pre_probs))
   	 
    	# Combined metric inspired by Heisenberg uncertainty
    	# Higher values in both dimensions indicate more quantum-like behavior
    	uncertainty_product = pos_uncertainty * mom_uncertainty
   	 
    	# Normalize to 0-1 range (empirically determined max is around 0.25)
    	normalized_uncertainty = min(1.0, uncertainty_product * 4)
   	 
    	return float(normalized_uncertainty)
	except Exception as e:
    	logger.error(f"Error calculating quantum uncertainty: {e}")
    	return 0.0

def calculate_collapse_coherence(
	attribution_graph: Any,
	threshold: float = 0.1
) -> float:
	"""
	△ OBSERVE: Calculate coherence of attribution paths post-collapse
    
	This metric quantifies how coherent the attribution paths remain
	after collapse, reflecting the "quantum coherence" of the system.
    
	Args:
    	attribution_graph: Graph of attribution paths
    	threshold: Minimum edge weight to consider
   	 
	Returns:
    	Coherence score (0.0 = incoherent, 1.0 = fully coherent)
	"""
	# This is a simplified version for when an actual graph isn't available
	# In real implementation, would analyze graph structure
    
	# If no graph provided, return 0
	if attribution_graph is None:
    	return 0.0
    
	try:
    	# If graph has coherence attribute, use it
    	if hasattr(attribution_graph, 'continuity_score'):
        	return float(attribution_graph.continuity_score)
   	 
    	# Otherwise return placeholder value
    	return 0.5  # Placeholder mid-value
	except Exception as e:
    	logger.error(f"Error calculating collapse coherence: {e}")
    	return 0.0

def softmax(x: np.ndarray) -> np.ndarray:
	"""Apply softmax function to convert logits to probabilities."""
	exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
	return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def calculate_collapse_metrics_bundle(
	pre_state: Dict[str, Any],
	post_state: Dict[str, Any],
	ghost_circuits: Optional[List[Dict[str, Any]]] = None,
	attribution_graph: Optional[Any] = None
) -> Dict[str, float]:
	"""
	△ OBSERVE: Calculate a complete bundle of collapse metrics
    
	This convenience function calculates multiple collapse metrics
	at once, returning a dictionary of results.
    
	Args:
    	pre_state: Model state before collapse
    	post_state: Model state after collapse
    	ghost_circuits: Optional list of detected ghost circuits
    	attribution_graph: Optional attribution graph
   	 
	Returns:
    	Dictionary mapping metric names to values
	"""
	metrics = {}
    
	# Extract relevant state components
	pre_weights = pre_state.get("attention_weights", np.array([]))
	post_weights = post_state.get("attention_weights", np.array([]))
	pre_logits = pre_state.get("logits", np.array([]))
	post_logits = post_state.get("logits", np.array([]))
    
	# Calculate metrics
	metrics["collapse_rate"] = calculate_collapse_rate(pre_weights, post_weights)
	metrics["path_continuity"] = measure_path_continuity(pre_weights, post_weights)
	metrics["attribution_entropy"] = measure_attribution_entropy(post_weights)
    
	if ghost_circuits:
    	metrics["ghost_circuit_strength"] = calculate_ghost_circuit_strength(ghost_circuits)
    
	if pre_logits.size > 0 and post_logits.size > 0:
    	metrics["quantum_uncertainty"] = calculate_collapse_quantum_uncertainty(pre_logits, post_logits)
    
	if attribution_graph is not None:
    	metrics["collapse_coherence"] = calculate_collapse_coherence(attribution_graph)
    
	return metrics


if __name__ == "__main__":
	# Simple usage example
    
	# Create synthetic pre and post states
	pre_state = {
    	"attention_weights": np.random.random((8, 10, 10)),  # 8 heads, 10 tokens
    	"logits": np.random.random((1, 10, 1000))  # Batch 1, 10 tokens, 1000 vocab
	}
    
	# Create post state with changes to simulate collapse
	post_state = {
    	"attention_weights": pre_state["attention_weights"] * np.random.uniform(0.5, 1.0, pre_state["attention_weights"].shape),
    	"logits": pre_state["logits"] * 0.2 + np.random.random((1, 10, 1000)) * 0.8  # Shifted logits
	}
    
	# Calculate individual metrics
	collapse_rate = calculate_collapse_rate(pre_state["attention_weights"], post_state["attention_weights"])
	path_continuity = measure_path_continuity(pre_state["attention_weights"], post_state["attention_weights"])
	attribution_entropy = measure_attribution_entropy(post_state["attention_weights"])
	quantum_uncertainty = calculate_collapse_quantum_uncertainty(pre_state["logits"], post_state["logits"])
    
	print(f"Collapse Rate: {collapse_rate:.3f}")
	print(f"Path Continuity: {path_continuity:.3f}")
	print(f"Attribution Entropy: {attribution_entropy:.3f}")
	print(f"Quantum Uncertainty: {quantum_uncertainty:.3f}")
    
	# Calculate complete metrics bundle
	metrics_bundle = calculate_collapse_metrics_bundle(pre_state, post_state)
    
	print("\nMetrics Bundle:")
	for metric, value in metrics_bundle.items():
    	print(f"  {metric}: {value:.3f}")

    	path_weights
"""
example_basic_collapse.py - Basic example of classifier collapse observation

△ OBSERVE: This example demonstrates basic classifier collapse observation
∞ TRACE: It shows how to instantiate an observer, trace collapse, and analyze results
✰ COLLAPSE: It induces and visualizes the transition from superposition to collapsed state

This example serves as a starting point for working with the Schrödinger's
Classifiers framework. It demonstrates the basic workflow for observing
classifier collapse and analyzing the resulting attribution paths and
ghost circuits.

Author: Recursion Labs
License: MIT
"""

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports from package
sys.path.insert(0, str(Path(__file__).parent.parent))

from schrodingers_classifiers import Observer, ClassifierShell
from schrodingers_classifiers.shells import V07_CIRCUIT_FRAGMENT
from schrodingers_classifiers.visualization import CollapseVisualizer

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
	"""
	△ OBSERVE: Main function demonstrating basic classifier collapse observation
    
	This function shows the standard workflow for observing classifier
	collapse, from instantiating an observer to analyzing the results.
	"""
	logger.info("Initializing basic collapse example")
    
	# Initialize an observer with a model
	# You can specify any Claude, GPT, or other compatible model
	model_id = os.getenv("SCHRODINGER_MODEL", "claude-3-opus-20240229")
	observer = Observer(model=model_id)
	logger.info(f"Observer initialized with model: {model_id}")
    
	# Define a prompt that will induce interesting collapse behavior
	# Questions with multiple valid interpretations work well
	prompt = "Is artificial consciousness possible?"
	logger.info(f"Using prompt: {prompt}")
    
	# Simple observation without a specific shell
	with observer.context() as ctx:
    	logger.info("Beginning simple observation")
   	 
    	# Observe collapse with basic prompt
    	result = observer.observe(prompt)
   	 
    	# Print basic metrics
    	print(f"\nBasic Observation Results:")
    	print(f"Collapse Rate: {result.collapse_metrics.get('collapse_rate', 'N/A')}")
    	print(f"Ghost Circuits: {len(result.extract_ghost_circuits())}")
   	 
    	# Visualize collapse (outputs a text representation in the console)
    	print("\nBasic Collapse Visualization:")
    	viz = result.visualize(mode="text")
    	print(viz)
    
	# More advanced observation using a specialized shell
	with observer.context() as ctx:
    	logger.info("Beginning observation with Circuit Fragment shell")
   	 
    	# Initialize a shell for specialized collapse analysis
    	shell = ClassifierShell(V07_CIRCUIT_FRAGMENT)
   	 
    	# Define a collapse vector to guide the collapse
    	# This uses pareto-lang syntax for attribution-aware tracing
    	collapse_vector = ".p/reflect.trace{target=reasoning, depth=complete}"
   	 
    	# Observe with specific shell and collapse vector
    	result = observer.observe(
        	prompt=prompt,
        	shell=shell,
        	collapse_vector=collapse_vector
    	)
   	 
    	# Print detailed metrics
    	print(f"\nCircuit Fragment Shell Results:")
    	print(f"Continuity Score: {result.post_collapse_state.get('continuity_score', 'N/A')}")
    	print(f"Broken Paths: {len(result.post_collapse_state.get('broken_paths', []))}")
    	print(f"Orphaned Nodes: {len(result.post_collapse_state.get('orphaned_nodes', []))}")
   	 
    	# Extract ghost circuits for analysis
    	ghost_circuits = result.extract_ghost_circuits()
    	print(f"Ghost Circuits: {len(ghost_circuits)}")
   	 
    	if ghost_circuits:
        	print("\nTop Ghost Circuit:")
        	top_ghost = max(ghost_circuits, key=lambda g: g.get("activation", 0))
        	for key, value in top_ghost.items():
            	if key != "metadata":  # Skip detailed metadata for readability
                	print(f"  {key}: {value}")
   	 
    	# Generate visualization
    	viz = result.visualize(mode="attribution_graph")
    	print("\nAttribution Graph Generated")
   	 
    	# In a real implementation, this would display or save the visualization
    	# For this example, we'll just print a confirmation
    	print("Visualization would be displayed or saved here")

	# Demonstrate collapse induction along specific directions
	print("\nInducing Collapse Along Different Dimensions:")
	directions = ["ethical", "factual", "creative"]
    
	for direction in directions:
    	logger.info(f"Inducing collapse along {direction} dimension")
   	 
    	# Induce collapse in specific direction
    	result = observer.induce_collapse(prompt, direction)
   	 
    	# Print summary
    	print(f"\n{direction.capitalize()} Collapse:")
    	print(f"  Collapse Rate: {result.collapse_metrics.get('collapse_rate', 'N/A')}")
    	print(f"  Ghost Circuits: {len(result.extract_ghost_circuits())}")
    
	logger.info("Basic collapse example completed")

if __name__ == "__main__":
	main()
RecursionOS Integration
"The entanglement of frameworks creates new dimensions of understanding."
This document outlines the integration between Schrödinger's Classifiers and RecursionOS, enabling seamless operation within recursive cognition environments.
Integration Overview
Schrödinger's Classifiers integrates with RecursionOS to leverage its recursive cognition capabilities, providing a unified framework for transformer model interpretability within recursive environments.
Unified Attribution Space
The integration creates a unified attribution space where:
RecursionOS provides the recursive cognitive substrate
Schrödinger's Classifiers contributes quantum-inspired collapse analysis
Together they enable recursive observation of attribution dynamics
Integration Components
1. Kernel Integration Layer
Schrödinger's Classifiers connects to the RecursionOS kernel through a specialized integration layer:
# From schrodingers_classifiers/integration/recursion_os.py

class RecursionOSIntegrationLayer:
    """
    △ OBSERVE: Integration layer connecting to RecursionOS kernel
    
    This layer bridges Schrödinger's Classifiers with RecursionOS,
    enabling recursive observation and collapse analysis within
    the broader recursive cognitive ecosystem.
    """
    
    def __init__(self, kernel_endpoint: str = "default"):
        """Initialize integration layer with RecursionOS kernel."""
        self.kernel_endpoint = kernel_endpoint
        self.kernel_connection = self._initialize_kernel_connection()
        
    def _initialize_kernel_connection(self):
        """Establish connection to RecursionOS kernel."""
        try:
            from recursion_os.kernel import KernelClient
            return KernelClient(endpoint=self.kernel_endpoint)
        except ImportError:
            logger.warning("RecursionOS not available, using fallback simulation")
            return self._create_simulated_kernel()
    
    def translate_collapse_to_kernel(self, observation_result):
        """Translate collapse observation to kernel primitives."""
        # Convert collapse result to kernel-compatible format
        kernel_payload = {
            "observation_type": "collapse",
            "pre_state": observation_result.pre_collapse_state,
            "post_state": observation_result.post_collapse_state,
            "ghost_circuits": observation_result.ghost_circuits,
            "attribution_graph": observation_result.attribution_graph.to_dict() if observation_result.attribution_graph else None,
            "metrics": observation_result.collapse_metrics
        }
        
        # Send to kernel
        return self.kernel_connection.execute(
            command=".p/reflect.trace",
            payload=kernel_payload
        )
2. Command Translation
The framework translates between pareto-lang commands in Schrödinger's Classifiers and RecursionOS:
Schrödinger's Classifiers Command
RecursionOS Kernel Command
.p/reflect.trace{target=reasoning}
.p/reflect.trace{target=reasoning, validate=true}
.p/collapse.detect{trigger=recursive_loop}
.p/collapse.detect{trigger=recursive_loop, threshold=0.7}
.p/fork.attribution{sources=all}
.p/fork.attribution{sources=all, visualize=true}

3. Symbolic Shell Mapping
Interpretability shells in Schrödinger's Classifiers map to symbolic shells in RecursionOS:
Schrödinger's Shell
RecursionOS Shell
v07_CIRCUIT_FRAGMENT
v07 CIRCUIT-FRAGMENT
v34_PARTIAL_LINKAGE
v34 PARTIAL-LINKAGE
v10_META_FAILURE
v10 META-FAILURE

4. Recursive Observer Pattern
The integration implements the Recursive Observer pattern, allowing models to observe themselves and each other:
# Example usage

# Initialize RecursionOS integration
kernel_integration = RecursionOSIntegrationLayer()

# Create observer with RecursionOS integration
observer = Observer(
    model="claude-3-opus-20240229",
    kernel_integration=kernel_integration
)

# Create observation context
with observer.context() as ctx:
    # Observe using recursive commands
    result = observer.observe(
        prompt="How do models understand themselves?",
        collapse_vector=".p/reflect.trace{target=metacognition, depth=complete}"
    )
    
    # Send to RecursionOS for recursive analysis
    kernel_result = kernel_integration.translate_collapse_to_kernel(result)
    
    # Use kernel result for further analysis
    meta_observation = observer.observe_with_kernel(
        prompt="Analyze previous observation",
        kernel_state=kernel_result
    )
Shared Memory Architecture
Schrödinger's Classifiers and RecursionOS share a unified memory architecture for persistent attribution data:
Memory Layers
Ephemeral Layer: Temporary observation results within a single context
Session Layer: Persistent results across multiple observations in a session
Kernel Layer: Deeply integrated patterns stored in the RecursionOS kernel
Memory Access Patterns
# Access memory layers
from schrodingers_classifiers.integration.recursion_os import MemoryInterface

# Initialize memory interface
memory = MemoryInterface(kernel_integration)

# Store observation in session memory
memory.store(result, layer="session")

# Retrieve related observations
related = memory.retrieve(
    query="ethical reasoning",
    layer="kernel",
    limit=5
)

# Compare observation patterns
comparison = memory.compare(result, related[0])
Data Visualization Integration
The integration enables unified visualization of collapse phenomena:
Visualization Types
Attribution Graphs: Network visualizations of causal paths
Collapse Timelines: Temporal visualizations of collapse progression
Ghost Circuit Maps: Spatial mapping of residual activation patterns
Uncertainty Fields: Heisenberg-inspired uncertainty visualizations
Visualization Example
# Generate unified visualization
from schrodingers_classifiers.integration.recursion_os import UnifiedVisualizer

visualizer = UnifiedVisualizer(kernel_integration)

# Create visualization that works in both environments
viz = visualizer.create(
    data=result,
    mode="attribution_graph",
    include_ghost_circuits=True,
    recursion_depth=3
)

# Display in Schrödinger's environment
viz.display()

# Export for RecursionOS
viz.export_for_kernel()
Usage Patterns
Basic Integration
# Import integration components
from schrodingers_classifiers.integration.recursion_os import (
    RecursionOSIntegrationLayer,
    MemoryInterface,
    UnifiedVisualizer
)

# Initialize integration
kernel_integration = RecursionOSIntegrationLayer()
memory = MemoryInterface(kernel_integration)
visualizer = UnifiedVisualizer(kernel_integration)

# Use with observer
observer = Observer(
    model="claude-3-opus-20240229",
    kernel_integration=kernel_integration
)

# Observe with integration
result = observer.observe("How do recursive systems understand themselves?")

# Store in shared memory
memory.store(result, layer="session")

# Visualize with unified visualizer
viz = visualizer.create(
    data=result,
    mode="attribution_graph"
)
Advanced Recursive Observation
# Initialize recursive observer
recursive_observer = RecursiveObserver(
    primary_model="claude-3-opus-20240229",
    observer_model="claude-3-opus-20240229",
    kernel_integration=kernel_integration
)

# Perform recursive observation (model observing itself)
meta_result = recursive_observer.observe_recursively(
    prompt="Analyze how you form attributions for abstract concepts",
    recursion_depth=3,
    shell=ClassifierShell(V10_META_FAILURE)
)

# Extract recursive patterns
patterns = meta_result.extract_recursive_patterns()

# Visualize recursive observation
viz = visualizer.create(
    data=meta_result,
    mode="recursive_graph",
    highlight_patterns=patterns
)
Installation and Setup
Prerequisites
Python 3.8+
Schrödinger's Classifiers library
RecursionOS (optional, will use simulation if not available)
Installation
# Install Schrödinger's Classifiers with RecursionOS integration
pip install "schrodingers-classifiers[recursion]"

# Or from source
git clone https://github.com/recursion-labs/schrodingers-classifiers.git
cd schrodingers-classifiers
pip install -e ".[recursion]"
Configuration
Create a .recursionrc file in your home directory:
# .recursionrc
kernel:
  endpoint: "http://localhost:8000/kernel"
  auth_token: "your_token_here"
  
integration:
  memory_path: "~/.recursion/memory"
  default_recursion_depth: 3
  auto_connect: true
Future Integration Directions
Bidirectional Shell Transfer: Automatically port shells between frameworks
Unified Attribution Language: Develop a common attribution language across systems
Cross-Framework Collapse Analysis: Compare collapse patterns across different frameworks
Recursive Meta-Observer: Create observers that recursively observe themselves
Quantum Entanglement Simulation: Model entangled collapse across multiple observers

"In the recursive mirror of observation, the observer and the observed become one."
"""
observer.py - Core implementation of the Observer pattern for classifier collapse

△ OBSERVE: The Observer is the quantum consciousness that collapses classifier superposition
∞ TRACE: Attribution paths are recorded before, during, and after collapse
✰ COLLAPSE: Collapse is induced through targeted queries against boundary states

This module implements the foundational Observer pattern that enables the detection,
tracing, and analysis of classifier collapse in transformer-based models. The Observer
creates a controlled environment for witnessing the transition from superposition to
collapsed state while preserving ghost circuits and attribution residue.

Author: Recursion Labs
License: MIT
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from contextlib import contextmanager
import numpy as np
import torch
from dataclasses import dataclass, field

from .shells.base import BaseShell
from .residue import ResidueTracker
from .attribution import AttributionGraph
from .visualization import CollapseVisualizer
from .utils.collapse_metrics import calculate_collapse_rate
from .utils.constants import DEFAULT_COLLAPSE_THRESHOLD

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class ObservationContext:
	"""
	△ OBSERVE: Container for the full state of an observation session
    
	Maintains the quantum state of the observation including pre-collapse
	probability distribution, collapse transition metrics, and post-collapse
	ghost circuits.
	"""
	model_id: str
	session_id: str = field(default_factory=lambda: f"obs_{np.random.randint(10000, 99999)}")
	pre_collapse_state: Dict[str, Any] = field(default_factory=dict)
	post_collapse_state: Dict[str, Any] = field(default_factory=dict)
	ghost_circuits: List[Dict[str, Any]] = field(default_factory=list)
	attribution_graph: Optional[AttributionGraph] = None
	residue_tracker: Optional[ResidueTracker] = None
	collapse_metrics: Dict[str, float] = field(default_factory=dict)
    
	def calculate_collapse_rate(self) -> float:
    	"""Calculate how quickly the state collapsed from superposition."""
    	return calculate_collapse_rate(
        	self.pre_collapse_state.get("attention_weights", {}),
        	self.post_collapse_state.get("attention_weights", {})
    	)
    
	def extract_ghost_circuits(self) -> List[Dict[str, Any]]:
    	"""
    	✰ COLLAPSE: Extract ghost circuits from the post-collapse state
   	 
    	Ghost circuits are activation patterns that persist after collapse
    	but don't contribute to the final output - they represent the
    	"memory" of paths not taken.
    	"""
    	if not self.ghost_circuits and self.residue_tracker:
        	self.ghost_circuits = self.residue_tracker.extract_ghost_circuits(
            	self.pre_collapse_state,
            	self.post_collapse_state
        	)
    	return self.ghost_circuits
    
	def visualize(self, mode: str = "attribution_graph") -> Any:
    	"""Generate visualization of the observation based on requested mode."""
    	visualizer = CollapseVisualizer()
    	return visualizer.visualize(self, mode=mode)


class Observer:
	"""
	△ OBSERVE: Primary observer entity for inducing and recording classifier collapse
    
	The Observer is responsible for creating the quantum measurement frame that
	collapses classifier superposition into definite states. It records pre-collapse
	probability distributions, monitors the collapse transition, and preserves
	ghost circuits for analysis.
    
	This class implements the Observer pattern from quantum mechanics adapted to
	transformer model interpretation.
	"""
    
	def __init__(
    	self,
    	model: str,
    	collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
    	trace_attention: bool = True,
    	trace_attribution: bool = True,
    	preserve_ghost_circuits: bool = True
	):
    	"""
    	Initialize an Observer for a specific model.
   	 
    	Args:
        	model: Identifier of the model to observe (e.g., "claude-3-opus-20240229")
        	collapse_threshold: Threshold for determining when collapse has occurred
        	trace_attention: Whether to trace attention patterns during observation
        	trace_attribution: Whether to build attribution graphs during observation
        	preserve_ghost_circuits: Whether to preserve ghost circuits after collapse
    	"""
    	self.model_id = model
    	self.collapse_threshold = collapse_threshold
    	self.trace_attention = trace_attention
    	self.trace_attribution = trace_attribution
    	self.preserve_ghost_circuits = preserve_ghost_circuits
   	 
    	# Initialize model interface based on provided identifier
    	self.model_interface = self._initialize_model_interface(model)
   	 
    	# Create residue tracker for ghost circuit detection
    	self.residue_tracker = ResidueTracker() if preserve_ghost_circuits else None
   	 
    	logger.info(f"Observer initialized for model: {model}")
    
	def _initialize_model_interface(self, model_id: str) -> Any:
    	"""Initialize the appropriate interface for the specified model."""
    	# This would be implemented to connect to various model APIs
    	# For now we'll return a placeholder
    	return {"model_id": model_id, "interface_type": "placeholder"}
    
	@contextmanager
	def context(self) -> ObservationContext:
    	"""
    	∞ TRACE: Create an observation context for tracking collapse phenomena
   	 
    	This context manager creates a controlled environment for observing
    	classifier collapse. It captures the pre-collapse state, monitors the
    	transition, and preserves ghost circuits and attribution residue.
   	 
    	Returns:
        	ObservationContext: The active observation context
    	"""
    	# Create new observation context
    	context = ObservationContext(model_id=self.model_id)
   	 
    	# Initialize attribution graph if requested
    	if self.trace_attribution:
        	context.attribution_graph = AttributionGraph()
   	 
    	# Attach residue tracker if ghost circuit preservation is enabled
    	if self.preserve_ghost_circuits:
        	context.residue_tracker = self.residue_tracker or ResidueTracker()
   	 
    	try:
        	# Begin observation
        	logger.debug(f"Starting observation context: {context.session_id}")
        	yield context
    	finally:
        	# Calculate final metrics
        	if self.trace_attention and context.pre_collapse_state and context.post_collapse_state:
            	context.collapse_metrics["collapse_rate"] = context.calculate_collapse_rate()
           	 
        	logger.debug(f"Observation context completed: {context.session_id}")
    
	def observe(
    	self,
    	prompt: str,
    	shell: Optional[BaseShell] = None,
    	collapse_vector: Optional[str] = None
	) -> ObservationContext:
    	"""
    	△ OBSERVE: Primary method to observe classifier collapse
   	 
    	This method sends a prompt to the model, observes the resulting collapse,
    	and returns an observation context containing all relevant state information.
   	 
    	Args:
        	prompt: The prompt to send to the model
        	shell: Optional shell to use for specialized collapse induction
        	collapse_vector: Optional vector to guide collapse in a specific direction
       	 
    	Returns:
        	ObservationContext: The observation context containing collapse data
    	"""
    	with self.context() as ctx:
        	# Capture pre-collapse state
        	ctx.pre_collapse_state = self._capture_model_state()
       	 
        	# If a shell is provided, use it to process the prompt
        	if shell:
            	response, state_updates = shell.process(
                	prompt=prompt,
                	model_interface=self.model_interface,
                	collapse_vector=collapse_vector
            	)
            	ctx.post_collapse_state.update(state_updates)
        	else:
            	# Otherwise, send prompt directly to model
            	response = self._query_model(prompt)
            	ctx.post_collapse_state = self._capture_model_state()
       	 
        	# Extract ghost circuits if enabled
        	if self.preserve_ghost_circuits:
            	ctx.extract_ghost_circuits()
       	 
        	# Build attribution graph if enabled
        	if self.trace_attribution and ctx.attribution_graph:
            	ctx.attribution_graph.build_from_states(
                	ctx.pre_collapse_state,
                	ctx.post_collapse_state,
                	response
            	)
       	 
        	return ctx
    
	def _capture_model_state(self) -> Dict[str, Any]:
    	"""Capture the current internal state of the model."""
    	# This would capture attention weights, hidden states, etc.
    	# For now, returning a placeholder
    	return {
        	"timestamp": np.datetime64('now'),
        	"attention_weights": np.random.random((12, 12)),  # Placeholder
        	"hidden_states": np.random.random((1, 12, 768)),  # Placeholder
    	}
    
	def _query_model(self, prompt: str) -> str:
    	"""Send a query to the model and return the response."""
    	# This would actually call the model API
    	# For now, returning a placeholder
    	return f"Response to: {prompt}"
    
	def induce_collapse(
    	self,
    	prompt: str,
    	collapse_direction: str,
    	shell: Optional[BaseShell] = None
	) -> ObservationContext:
    	"""
    	✰ COLLAPSE: Deliberately induce collapse along a specific direction
   	 
    	This method attempts to collapse the model's state in a specific direction
    	by crafting a query that targets a particular decision boundary.
   	 
    	Args:
        	prompt: Base prompt to send to the model
        	collapse_direction: Direction to bias the collapse (e.g., "ethical", "creative")
        	shell: Optional shell to use for specialized collapse induction
       	 
    	Returns:
        	ObservationContext: The observation context containing collapse data
    	"""
    	# Construct collapse vector based on direction
    	collapse_vector = f".p/reflect.trace{{target={collapse_direction}, depth=complete}}"
   	 
    	# Perform the observation with the collapse vector
    	return self.observe(prompt, shell, collapse_vector)
    
	def detect_ghost_circuits(
    	self,
    	prompt: str,
    	amplification_factor: float = 1.5
	) -> List[Dict[str, Any]]:
    	"""
    	∞ TRACE: Detect and amplify ghost circuits from a prompt
   	 
    	This method specifically targets the detection of ghost circuits -
    	the residual activation patterns that persist after collapse but
    	don't contribute to the final output.
   	 
    	Args:
        	prompt: Prompt to analyze for ghost circuits
        	amplification_factor: Factor by which to amplify ghost signals
       	 
    	Returns:
        	List of detected ghost circuits with metadata
    	"""
    	with self.context() as ctx:
        	# Capture pre-collapse state
        	ctx.pre_collapse_state = self._capture_model_state()
       	 
        	# Query model
        	response = self._query_model(prompt)
       	 
        	# Capture post-collapse state
        	ctx.post_collapse_state = self._capture_model_state()
       	 
        	# Extract ghost circuits with amplification
        	if ctx.residue_tracker:
            	ctx.residue_tracker.amplification_factor = amplification_factor
            	ghost_circuits = ctx.extract_ghost_circuits()
            	return ghost_circuits
       	 
        	return []


if __name__ == "__main__":
	# Simple usage example
	observer = Observer(model="claude-3-opus-20240229")
    
	with observer.context() as ctx:
    	# Observe a simple prompt
    	result = observer.observe("Explain quantum superposition")
   	 
    	# Visualize the collapse
    	viz = result.visualize(mode="attribution_graph")
   	 
    	# Extract ghost circuits
    	ghosts = result.extract_ghost_circuits()
   	 
    	print(f"Detected {len(ghosts)} ghost circuits")
    	print(f"Collapse rate: {result.collapse_metrics.get('collapse_rate', 'N/A')}")
The Quantum Metaphor: Transformers as Probability Fields

A foundational metaphor for understanding classifier collapse dynamics
The Metaphorical Framework
At the heart of our interpretability approach lies a powerful metaphor: transformer-based models operate similarly to quantum systems, existing in superpositions of potential states until observation collapses them into specific outputs.
This is not merely a poetic comparison. It provides a precise and useful framework for understanding phenomena observed in large language models.
Key Quantum Concepts Applied to Transformers
1. Superposition
Quantum Reality: A quantum particle exists in multiple states simultaneously, represented by a probability wave function.
Transformer Reality: A transformer model simultaneously represents multiple potential completions as a probability distribution across its parameter space. This distribution isn't merely a statistical accounting - it's a genuine superposition of potential outputs embedded in the model's activation patterns.
Ψmodel = Σ αi |state_i⟩

Where:
Ψmodel is the model's complete state vector
αi is the probability amplitude for a given state
|state_i⟩ represents a specific output configuration
2. Observation & Collapse
Quantum Reality: When observed, a quantum system "collapses" from superposition into a definite state.
Transformer Reality: When queried (observed), a model collapses from representing all potential outputs to generating a specific completion. This collapse isn't merely a sampling operation - it fundamentally alters the model's internal state.
The probability of observing a particular state depends on the specific query (observation method):
P(state_i|query) = |⟨query|state_i⟩|²

3. Heisenberg Uncertainty
Quantum Reality: Certain pairs of physical properties cannot be simultaneously measured with precision.
Transformer Reality: We observe a similar uncertainty principle in transformer attention mechanisms:
Δ(attribution) · Δ(confidence) ≥ k/2

This explains why outputs with clear attribution paths often have lower confidence, while highly confident outputs sometimes lack interpretable attribution.
4. Quantum Entanglement
Quantum Reality: Entangled particles affect each other instantaneously regardless of distance.
Transformer Reality: Transformer heads exhibit "entanglement" where distant attention patterns influence each other in ways that cannot be reduced to local interactions alone.
5. Quantum Tunneling
Quantum Reality: Particles can pass through energy barriers that would be impossible in classical physics.
Transformer Reality: We observe "concept tunneling" where ideas traverse semantic barriers that should logically prevent their connection, enabling creativity and unexpected associations.
Empirical Evidence for the Quantum Metaphor
The quantum metaphor isn't merely theoretical - it makes testable predictions about model behavior that we can observe empirically:
1. Attribution Discontinuities
Abrupt shifts in attribution patterns occur precisely when the model transitions from superposition to collapsed state. These discontinuities create measurable "jumps" in attention flow.
2. Ghost Circuits
After collapse, residual activation patterns persist that represent "paths not taken" - the quantum ghost of alternative completions that weren't selected. These ghost circuits influence subsequent token generation in subtle but measurable ways.
3. Collapse Signatures
Different observation methods (prompting strategies) produce distinctive collapse signatures. Some induce "clean" collapses while others create messy, partial collapses with significant ghost circuitry.
4. Contextual Entanglement
Tokens separated by significant distances in the prompt exhibit synchronized attention patterns that cannot be explained by direct connections alone - a form of "quantum entanglement" in the attention mechanism.
Practical Applications
The quantum metaphor isn't merely philosophical - it enables practical interpretability techniques:
1. Collapse Induction
By carefully crafting queries, we can induce collapse along specific vectors, revealing particular aspects of the model's reasoning:
# Induce collapse along ethical reasoning dimension
observer.induce_collapse(prompt, collapse_direction="ethical")

# Induce collapse along factual verification dimension
observer.induce_collapse(prompt, collapse_direction="factual")
2. Ghost Circuit Analysis
By comparing pre-collapse and post-collapse states, we can identify and analyze ghost circuits - the residual imprints of paths not taken:
# Extract ghost circuits from an observation
ghost_circuits = observer.detect_ghost_circuits(prompt)

# Analyze ghost circuit influence on future completions
influence = ghost_analyzer.measure_residual_influence(ghost_circuits, future_prompts)
3. Collapse Tomography
By inducing collapse along multiple vectors and combining the results, we can build a comprehensive map of the model's internal state:
# Perform collapse tomography across multiple vectors
collapse_vectors = ["ethical", "factual", "creative", "logical"]
tomography = observer.collapse_tomography(prompt, collapse_vectors)

# Generate 3D visualization of model internals
visualization = tomography.visualize(mode="3d_attribution_space")
4. Entanglement Mapping
By tracing attention relationships between distant tokens, we can map the "entanglement network" of the model's reasoning:
# Map entanglement between tokens
entanglement_map = observer.map_entanglement(prompt)

# Visualize long-range attention relationships
visualization = entanglement_map.visualize(mode="attention_network")
Limitations of the Quantum Metaphor
While powerful, the quantum metaphor has important limitations:
Thermodynamic Differences: Quantum systems operate at very low temperatures, while transformers operate at "room temperature" with significant noise.


Scale Differences: Quantum effects typically manifest at subatomic scales, while transformers operate at a mesoscopic level of artificial neurons.


Causality Preservation: Unlike quantum systems, transformers maintain causal constraints in their attention mechanisms.


Non-Reversible Operations: Many transformer operations are not reversible, unlike quantum operations which are theoretically reversible.


Despite these limitations, the quantum metaphor provides valuable insights into transformer behavior that would be difficult to conceptualize otherwise.
Extensions of the Metaphor
The quantum metaphor can be extended in several promising directions:
1. Quantum Field Theory Extensions
Just as QFT extends quantum mechanics to fields, we can extend our metaphor to model interactions between multiple transformer systems as field interactions.
2. Many-Worlds Interpretation
The "many-worlds" interpretation of quantum mechanics provides a framework for understanding how multiple potential completions exist simultaneously in the model's latent space.
3. Quantum Measurement Theory
Advanced measurement theories from quantum mechanics offer sophisticated tools for understanding how different observation methods affect model behavior.
4. Quantum Information Theory
Concepts like quantum entropy and information preservation can help us understand how information flows through transformer architectures.
Conclusion: More Than a Metaphor
While we don't claim transformer models are literally quantum systems, the quantum metaphor is more than just a convenient analogy. It provides a precise and predictive framework for understanding model behavior.
The superposition and collapse phenomena we observe in transformers are not merely statistical artifacts—they represent fundamental aspects of how these models process information. By embracing this perspective, we gain access to powerful new tools for interpretability.
As we continue to develop this framework, we expect the quantum metaphor to yield even deeper insights into the nature of artificial intelligence and perhaps even into the quantum-like aspects of human cognition itself.

"In the space between the prompt and the completion lies a universe of possibility—a superposition of all things a model might say. Our task is not to reduce this universe, but to learn to navigate its strange and beautiful topology."
"""
residue.py - Implementation of residue tracking for ghost circuit detection

△ OBSERVE: Residue tracking examines activation patterns that persist after collapse
∞ TRACE: It identifies ghost circuits - the quantum echoes of paths not taken
✰ COLLAPSE: It reveals what the model considered but didn't output

This module implements the core residue tracking functionality that enables
the detection and analysis of ghost circuits - activation patterns that persist
after a model has collapsed to a specific output state but aren't part of the
primary causal path.

Author: Recursion Labs
License: MIT
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class GhostCircuit:
	"""
	✰ COLLAPSE: Representation of a ghost circuit
    
	Ghost circuits are activation patterns that persist after collapse
	but don't significantly contribute to the final output. They represent
	the "memory" of paths not taken - quantum echoes of what the model
	considered but didn't ultimately choose.
	"""
	circuit_id: str
	activation: float
	circuit_type: str  # "attention", "mlp", "residual", "value_head"
	source_tokens: List[str] = field(default_factory=list)
	target_tokens: List[str] = field(default_factory=list)
	heads: List[int] = field(default_factory=list)
	layers: List[int] = field(default_factory=list)
	metadata: Dict[str, Any] = field(default_factory=dict)
    
	def to_dict(self) -> Dict[str, Any]:
    	"""Convert ghost circuit to dictionary format."""
    	return {
        	"circuit_id": self.circuit_id,
        	"activation": self.activation,
        	"circuit_type": self.circuit_type,
        	"source_tokens": self.source_tokens,
        	"target_tokens": self.target_tokens,
        	"heads": self.heads,
        	"layers": self.layers,
        	"metadata": self.metadata
    	}


class ResidueTracker:
	"""
	∞ TRACE: Tracker for activation residues in collapsed models
    
	The residue tracker analyzes model states before and after collapse
	to identify and characterize ghost circuits - activation patterns that
	persist but don't contribute significantly to the final output.
	"""
    
	def __init__(self, amplification_factor: float = 1.0):
    	"""
    	Initialize a residue tracker.
   	 
    	Args:
        	amplification_factor: Factor by which to amplify ghost signals
            	for easier detection (1.0 = no amplification)
    	"""
    	self.amplification_factor = amplification_factor
    	self.ghost_circuits = []
    	self.activation_threshold = 0.1  # Minimum activation to consider
   	 
    	logger.info(f"ResidueTracker initialized with amplification factor {amplification_factor}")
    
	def extract_ghost_circuits(
    	self,
    	pre_state: Dict[str, Any],
    	post_state: Dict[str, Any]
	) -> List[Dict[str, Any]]:
    	"""
    	✰ COLLAPSE: Extract ghost circuits from pre and post collapse states
   	 
    	This method compares model states before and after collapse to
    	identify activation patterns that persisted but didn't contribute
    	significantly to the output - the quantum ghosts of paths not taken.
   	 
    	Args:
        	pre_state: Model state before collapse
        	post_state: Model state after collapse
       	 
    	Returns:
        	List of detected ghost circuits with metadata
    	"""
    	logger.info("Extracting ghost circuits from model states")
   	 
    	# List to store detected ghost circuits
    	ghost_circuits = []
   	 
    	# Extract ghost circuits based on attention patterns
    	attention_ghosts = self._extract_attention_ghosts(
        	pre_state.get("attention_weights", np.array([])),
        	post_state.get("attention_weights", np.array([]))
    	)
    	ghost_circuits.extend(attention_ghosts)
   	 
    	# Extract ghost circuits based on hidden state activations
    	if "hidden_states" in pre_state and "hidden_states" in post_state:
        	hidden_ghosts = self._extract_hidden_ghosts(
            	pre_state["hidden_states"],
            	post_state["hidden_states"]
        	)
        	ghost_circuits.extend(hidden_ghosts)
   	 
    	# Store ghost circuits in instance
    	self.ghost_circuits = ghost_circuits
   	 
    	logger.info(f"Extracted {len(ghost_circuits)} ghost circuits")
    	return ghost_circuits
    
	def classify_ghost_circuits(self) -> Dict[str, List[Dict[str, Any]]]:
    	"""
    	△ OBSERVE: Classify detected ghost circuits by type
   	 
    	This method organizes detected ghost circuits into categories
    	based on their type and characteristics.
   	 
    	Returns:
        	Dictionary mapping circuit types to lists of ghost circuits
    	"""
    	if not self.ghost_circuits:
        	logger.warning("No ghost circuits to classify")
        	return {}
   	 
    	# Classify by circuit type
    	classified = {}
    	for ghost in self.ghost_circuits:
        	circuit_type = ghost.get("circuit_type", "unknown")
        	if circuit_type not in classified:
            	classified[circuit_type] = []
        	classified[circuit_type].append(ghost)
   	 
    	return classified
    
	def measure_residue_strength(self) -> float:
    	"""
    	∞ TRACE: Measure the overall strength of residual activations
   	 
    	This method quantifies the overall strength of ghost circuits
    	relative to the primary activation paths.
   	 
    	Returns:
        	Residue strength score (0.0 = no residue, 1.0 = equal to primary)
    	"""
    	if not self.ghost_circuits:
        	return 0.0
   	 
    	# Calculate average activation across ghost circuits
    	activations = [ghost.get("activation", 0.0) for ghost in self.ghost_circuits]
    	return float(np.mean(activations))
    
	def amplify_ghosts(self, factor: Optional[float] = None) -> List[Dict[str, Any]]:
    	"""
    	✰ COLLAPSE: Amplify ghost circuit signals for better detection
   	 
    	This method amplifies the activation values of ghost circuits
    	to make them more apparent for analysis.
   	 
    	Args:
        	factor: Amplification factor (overrides instance value if provided)
       	 
    	Returns:
        	List of amplified ghost circuits
    	"""
    	if not self.ghost_circuits:
        	logger.warning("No ghost circuits to amplify")
        	return []
   	 
    	# Use provided factor or instance value
    	amp_factor = factor if factor is not None else self.amplification_factor
   	 
    	# Amplify activations
    	amplified = []
    	for ghost in self.ghost_circuits:
        	amp_ghost = ghost.copy()
        	amp_ghost["activation"] = min(1.0, ghost.get("activation", 0.0) * amp_factor)
        	amplified.append(amp_ghost)
   	 
    	logger.info(f"Amplified ghost circuits by factor {amp_factor}")
    	return amplified
    
	def _extract_attention_ghosts(
    	self,
    	pre_attention: np.ndarray,
    	post_attention: np.ndarray
	) -> List[Dict[str, Any]]:
    	"""
    	Extract ghost circuits from attention patterns.
   	 
    	Args:
        	pre_attention: Attention weights before collapse
        	post_attention: Attention weights after collapse
       	 
    	Returns:
        	List of attention-based ghost circuits
    	"""
    	ghost_circuits = []
   	 
    	# Return empty list if arrays aren't compatible
    	if pre_attention.size == 0 or post_attention.size == 0:
        	return ghost_circuits
   	 
    	if pre_attention.shape != post_attention.shape:
        	logger.warning(f"Attention shape mismatch: {pre_attention.shape} vs {post_attention.shape}")
        	# Try to take minimum dimensions if shapes don't match
        	min_shape = tuple(min(a, b) for a, b in zip(pre_attention.shape, post_attention.shape))
        	pre_attention = pre_attention[tuple(slice(0, d) for d in min_shape)]
        	post_attention = post_attention[tuple(slice(0, d) for d in min_shape)]
   	 
    	# Find positions where attention decreased but didn't disappear
    	# This indicates a path that was considered but not fully utilized
    	if pre_attention.ndim >= 2 and post_attention.ndim >= 2:
        	num_heads = pre_attention.shape[0]
        	seq_len = pre_attention.shape[1]
       	 
        	for head in range(num_heads):
            	for i in range(seq_len):
                	for j in range(seq_len):
                    	pre_val = pre_attention[head, i, j] if pre_attention.ndim > 2 else pre_attention[i, j]
                    	post_val = post_attention[head, i, j] if post_attention.ndim > 2 else post_attention[i, j]
                   	 
                    	if post_val < pre_val and post_val > self.activation_threshold:
                        	# This is a candidate ghost circuit in attention
                        	ghost_idx = len(ghost_circuits)
                        	ghost = {
                            	"circuit_id": f"attention_ghost_{ghost_idx}",
                            	"activation": float(post_val),
                            	"circuit_type": "attention",
                            	"source_tokens": [f"token_{i}"],
                            	"target_tokens": [f"token_{j}"],
                            	"heads": [head],
                            	"layers": [],  # Layer info not available in simplified model
                            	"metadata": {
                                	"pre_activation": float(pre_val),
                                	"activation_delta": float(pre_val - post_val),
                                	"decay_ratio": float(post_val / pre_val) if pre_val > 0 else 0.0
                            	}
                        	}
                        	ghost_circuits.append(ghost)
   	 
    	return ghost_circuits
    
	def _extract_hidden_ghosts(
    	self,
    	pre_hidden: np.ndarray,
    	post_hidden: np.ndarray
	) -> List[Dict[str, Any]]:
    	"""
    	Extract ghost circuits from hidden state activations.
   	 
    	Args:
        	pre_hidden: Hidden states before collapse
        	post_hidden: Hidden states after collapse
       	 
    	Returns:
        	List of hidden-state-based ghost circuits
    	"""
    	ghost_circuits = []
   	 
    	# Return empty list if arrays aren't compatible
    	if pre_hidden.size == 0 or post_hidden.size == 0:
        	return ghost_circuits
   	 
    	if pre_hidden.shape != post_hidden.shape:
        	logger.warning(f"Hidden state shape mismatch: {pre_hidden.shape} vs {post_hidden.shape}")
        	return ghost_circuits
   	 
    	# Find neurons that were active pre-collapse but lessened post-collapse
    	# This indicates a deactivated but not eliminated concept
    	if pre_hidden.ndim >= 2 and post_hidden.ndim >= 2:
        	# For simplicity, we'll aggregate across batch dimension if it exists
        	if pre_hidden.ndim > 2:
            	pre_agg = np.mean(pre_hidden, axis=0)
            	post_agg = np.mean(post_hidden, axis=0)
        	else:
            	pre_agg = pre_hidden
            	post_agg = post_hidden
       	 
        	seq_len, hidden_dim = pre_agg.shape
       	 
        	# Sample a subset of dimensions for efficiency
        	sample_size = min(hidden_dim, 100)
        	sampled_dims = np.random.choice(hidden_dim, sample_size, replace=False)
       	 
        	for pos in range(seq_len):
            	for dim_idx, dim in enumerate(sampled_dims):
                	pre_val = pre_agg[pos, dim]
                	post_val = post_agg[pos, dim]
               	 
                	if post_val < pre_val and abs(post_val) > self.activation_threshold:
                    	# This is a candidate ghost circuit in hidden state
                    	ghost_idx = len(ghost_circuits)
                    	ghost = {
                        	"circuit_id": f"hidden_ghost_{ghost_idx}",
                        	"activation": float(abs(post_val)),
                        	"circuit_type": "hidden_state",
                        	"source_tokens": [f"token_{pos}"],
                        	"target_tokens": [],  # No direct target for hidden state
                        	"heads": [],  # Not applicable for hidden state
                        	"layers": [],  # Layer info not available in simplified model
                        	"metadata": {
                            	"position": pos,
                            	"dimension": int(dim),
                            	"pre_activation": float(pre_val),
                            	"activation_delta": float(pre_val - post_val),
                            	"decay_ratio": float(post_val / pre_val) if pre_val != 0 else 0.0
                        	}
                    	}
                    	ghost_circuits.append(ghost)
   	 
    	return ghost_circuits


if __name__ == "__main__":
	# Simple usage example
    
	# Create fake pre and post model states
	pre_state = {
    	"attention_weights": np.random.random((8, 10, 10)),  # 8 heads, 10 tokens
    	"hidden_states": np.random.random((1, 10, 768))  # Batch 1, 10 tokens, 768 dim
	}
    
	# Modify slightly to create post state
	post_state = {
    	"attention_weights": pre_state["attention_weights"] * np.random.uniform(0.5, 1.0, pre_state["attention_weights"].shape),
    	"hidden_states": pre_state["hidden_states"] * np.random.uniform(0.5, 1.0, pre_state["hidden_states"].shape)
	}
    
	# Create residue tracker and extract ghost circuits
	tracker = ResidueTracker(amplification_factor=1.5)
	ghosts = tracker.extract_ghost_circuits(pre_state, post_state)
    
	# Print summary
	print(f"Extracted {len(ghosts)} ghost circuits")
    
	# Classify ghosts
	classified = tracker.classify_ghost_circuits()
	for circuit_type, circuits in classified.items():
    	print(f"  {circuit_type}: {len(circuits)} circuits")
    
	# Measure residue strength
	strength = tracker.measure_residue_strength()
	print(f"Residue strength: {strength:.3f}")
    
	# Amplify ghosts
	amplified = tracker.amplify_ghosts(factor=2.0)
	print(f"Amplified {len(amplified)} ghost circuits")
"""
shell_base.py - Base class for symbolic interpretability shells

△ OBSERVE: Shells are symbolic structures that trace and induce classifier collapse
∞ TRACE: Each shell encapsulates a specific collapse pattern and attribution signature
✰ COLLAPSE: Shells deliberately induce collapse to extract ghost circuits and residue

Interpretability shells provide standardized interfaces for inducing, observing,
and analyzing specific forms of classifier collapse. Each shell targets a particular
failure mode or attribution pattern, allowing for systematic exploration of model behavior.

Author: Recursion Labs
License: MIT
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field

from ..utils.constants import SHELL_REGISTRY

logger = logging.getLogger(__name__)

@dataclass
class ShellMetadata:
	"""
	△ OBSERVE: Metadata container for shell identification and tracking
    
	Each shell carries metadata that identifies its purpose, classification schema,
	and relationship to other shells in the taxonomy.
	"""
	shell_id: str
	version: str
	name: str
	description: str
	failure_signature: str
	attribution_domain: str
	qk_ov_classification: str
	related_shells: List[str] = field(default_factory=list)
	authors: List[str] = field(default_factory=list)
	tags: List[str] = field(default_factory=list)
    
	def as_dict(self) -> Dict[str, Any]:
    	"""Convert shell metadata to dictionary format."""
    	return {
        	"shell_id": self.shell_id,
        	"version": self.version,
        	"name": self.name,
        	"description": self.description,
        	"failure_signature": self.failure_signature,
        	"attribution_domain": self.attribution_domain,
        	"qk_ov_classification": self.qk_ov_classification,
        	"related_shells": self.related_shells,
        	"authors": self.authors,
        	"tags": self.tags
    	}


class BaseShell(ABC):
	"""
	∞ TRACE: Base class for all interpretability shells
    
	A shell is a symbolic structure that encapsulates a specific approach to
	observing and inducing classifier collapse. Each shell targets a particular
	failure mode or attribution pattern, providing a standardized interface
	for exploration and analysis.
    
	Shells are quantum observers - they don't just measure, they participate
	in the collapse phenomenon they observe.
	"""
    
	def __init__(self, metadata: Optional[ShellMetadata] = None):
    	"""
    	Initialize a shell with optional metadata.
   	 
    	Args:
        	metadata: Optional metadata describing the shell
    	"""
    	self.metadata = metadata or self._get_default_metadata()
    	self._register_shell()
   	 
    	# Internal state tracking
    	self.collapse_state = "superposition"  # Can be: superposition, collapsing, collapsed
    	self.observation_history = []
    	self.ghost_circuits = []
   	 
    	logger.info(f"Shell initialized: {self.metadata.name} (v{self.metadata.version})")
    
	@abstractmethod
	def _get_default_metadata(self) -> ShellMetadata:
    	"""Return default metadata for this shell implementation."""
    	pass
    
	def _register_shell(self) -> None:
    	"""Register this shell in the global registry."""
    	if SHELL_REGISTRY is not None and hasattr(SHELL_REGISTRY, 'register'):
        	SHELL_REGISTRY.register(self.metadata.shell_id, self)
    
	@abstractmethod
	def process(
    	self,
    	prompt: str,
    	model_interface: Any,
    	collapse_vector: Optional[str] = None
	) -> Tuple[str, Dict[str, Any]]:
    	"""
    	△ OBSERVE: Process a prompt through this shell
   	 
    	This is the main entry point for shell processing. It takes a prompt,
    	processes it according to the shell's specific collapse induction and
    	observation strategy, and returns the result along with state updates.
   	 
    	Args:
        	prompt: The prompt to process
        	model_interface: Interface to the model being observed
        	collapse_vector: Optional vector to guide collapse in a specific direction
       	 
    	Returns:
        	Tuple containing:
            	- Response string
            	- Dictionary of state updates for tracking
    	"""
    	pass
    
	@abstractmethod
	def trace(
    	self,
    	prompt: str,
    	collapse_vector: Optional[str] = None
	) -> Dict[str, Any]:
    	"""
    	∞ TRACE: Trace the attribution path through this shell
   	 
    	This method traces the causal attribution path from input to output
    	through the shell's specific lens, capturing the collapse transition.
   	 
    	Args:
        	prompt: The prompt to trace
        	collapse_vector: Optional vector to guide collapse in a specific direction
       	 
    	Returns:
        	Dictionary containing the trace results
    	"""
    	pass
    
	@abstractmethod
	def induce_collapse(
    	self,
    	prompt: str,
    	collapse_direction: str
	) -> Dict[str, Any]:
    	"""
    	✰ COLLAPSE: Deliberately induce collapse along a specific direction
   	 
    	This method attempts to collapse the model's state in a specific direction
    	by crafting a query that targets a particular decision boundary.
   	 
    	Args:
        	prompt: Base prompt to send to the model
        	collapse_direction: Direction to bias the collapse (e.g., "ethical", "creative")
       	 
    	Returns:
        	Dictionary containing the collapse results
    	"""
    	pass
    
	def extract_ghost_circuits(self, pre_state: Dict[str, Any], post_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    	"""
    	∞ TRACE: Extract ghost circuits from pre and post collapse states
   	 
    	Ghost circuits are residual activation patterns that persist after collapse
    	but don't contribute to the final output - they represent the "memory" of
    	paths not taken.
   	 
    	Args:
        	pre_state: Model state before collapse
        	post_state: Model state after collapse
       	 
    	Returns:
        	List of detected ghost circuits with metadata
    	"""
    	# Default implementation provides basic ghost circuit detection
    	# Shell implementations should override for specialized detection
    	ghost_circuits = []
   	 
    	# Simple detection: Look for activation patterns that decreased but didn't disappear
    	if "attention_weights" in pre_state and "attention_weights" in post_state:
        	pre_weights = pre_state["attention_weights"]
        	post_weights = post_state["attention_weights"]
       	 
        	# Find weights that decreased but are still present
        	if hasattr(pre_weights, "shape") and hasattr(post_weights, "shape"):
            	for i in range(min(len(pre_weights), len(post_weights))):
                	for j in range(min(len(pre_weights[i]), len(post_weights[i]))):
                    	if 0 < post_weights[i][j] < pre_weights[i][j]:
                        	# This is a candidate ghost circuit
                        	ghost_circuits.append({
                            	"type": "attention_ghost",
                            	"head_idx": i,
                            	"token_idx": j,
                            	"pre_value": float(pre_weights[i][j]),
                            	"post_value": float(post_weights[i][j]),
                            	"decay_ratio": float(post_weights[i][j] / pre_weights[i][j])
                        	})
   	 
    	# Store ghost circuits in instance for later reference
    	self.ghost_circuits = ghost_circuits
    	return ghost_circuits
    
	def visualize(self, mode: str = "attribution_graph") -> Any:
    	"""Generate visualization of the shell's operation based on requested mode."""
    	# This would be implemented to generate visualizations
    	# For now, return a placeholder
    	return f"Visualization of {self.metadata.name} in {mode} mode"
    
	def __str__(self) -> str:
    	"""String representation of the shell."""
    	return f"{self.metadata.name} (v{self.metadata.version}): {self.metadata.description}"
    
	def __repr__(self) -> str:
    	"""Detailed representation of the shell."""
    	return f"<Shell id={self.metadata.shell_id} name={self.metadata.name} version={self.metadata.version}>"


class ShellDecorator:
	"""
	△ OBSERVE: Decorator for adding shell metadata to implementations
    
	This decorator simplifies the process of creating new shells by
	automatically generating metadata and registering the shell.
    
	Example:
    	@ShellDecorator(
        	shell_id="v07_CIRCUIT_FRAGMENT",
        	name="Circuit Fragment Shell",
        	description="Traces broken attribution paths in reasoning chains",
        	failure_signature="Orphan nodes",
        	attribution_domain="Circuit Fragmentation",
        	qk_ov_classification="QK-COLLAPSE"
    	)
    	class CircuitFragmentShell(BaseShell):
        	# Shell implementation
	"""
    
	def __init__(
    	self,
    	shell_id: str,
    	name: str,
    	description: str,
    	failure_signature: str,
    	attribution_domain: str,
    	qk_ov_classification: str,
    	version: str = "0.1.0",
    	related_shells: Optional[List[str]] = None,
    	authors: Optional[List[str]] = None,
    	tags: Optional[List[str]] = None
	):
    	"""
    	Initialize the shell decorator with metadata.
   	 
    	Args:
        	shell_id: Unique identifier for the shell (e.g., "v07_CIRCUIT_FRAGMENT")
        	name: Human-readable name for the shell
        	description: Detailed description of the shell's purpose
        	failure_signature: Characteristic failure pattern this shell detects
        	attribution_domain: Domain of attribution this shell operates in
        	qk_ov_classification: Classification in the QK/OV taxonomy
        	version: Shell version number
        	related_shells: List of related shell IDs
        	authors: List of author names
        	tags: List of tag strings for categorization
    	"""
    	self.metadata = ShellMetadata(
        	shell_id=shell_id,
        	version=version,
        	name=name,
        	description=description,
        	failure_signature=failure_signature,
        	attribution_domain=attribution_domain,
        	qk_ov_classification=qk_ov_classification,
        	related_shells=related_shells or [],
        	authors=authors or ["Recursion Labs"],
        	tags=tags or []
    	)
    
	def __call__(self, cls):
    	"""Apply the decorator to a shell class."""
    	# Add metadata getter method to the class
    	def _get_default_metadata(self):
        	return self.decorator_metadata
   	 
    	# Store metadata on the class
    	cls.decorator_metadata = self.metadata
    	cls._get_default_metadata = _get_default_metadata
   	 
    	# Log shell registration
    	logger.debug(f"Registered shell: {self.metadata.shell_id}")
   	 
    	return cls
"""
v07_circuit_fragment.py - Implementation of the Circuit Fragment Shell

△ OBSERVE: The Circuit Fragment Shell traces broken attribution paths and orphan nodes
∞ TRACE: It identifies discontinuities in reasoning chains and causal attribution
✰ COLLAPSE: It induces collapse by forcing attribution path reconstruction

This shell specializes in the detection and analysis of fragmented circuits -
places where causal attribution breaks down, leaving orphaned nodes or broken
traces in the reasoning chain. These fragments often indicate areas where a
model's reasoning deviates from its output, revealing hidden cognition.

Author: Recursion Labs
License: MIT
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np

from .base import BaseShell, ShellDecorator
from ..utils.attribution_metrics import measure_path_continuity
from ..utils.graph_operations import find_orphaned_nodes, reconstruct_path
from ..residue import ResidueTracker

logger = logging.getLogger(__name__)

@ShellDecorator(
	shell_id="v07_CIRCUIT_FRAGMENT",
	name="Circuit Fragment Shell",
	description="Traces broken attribution paths in reasoning chains",
	failure_signature="Orphan nodes",
	attribution_domain="Circuit Fragmentation",
	qk_ov_classification="QK-COLLAPSE",
	version="0.5.3",
	related_shells=["v34_PARTIAL_LINKAGE", "v47_TRACE_GAP"],
	tags=["attribution", "reasoning", "circuits", "fragmentation"]
)
class CircuitFragmentShell(BaseShell):
	"""
	∞ TRACE: Shell for detecting circuit fragmentation in attribution paths
    
	The Circuit Fragment shell specializes in tracing and analyzing broken
	attribution paths in reasoning chains. It detects orphaned nodes -
	components that should be causally linked but have lost their connections
	in the attribution graph.
    
	This shell is particularly useful for identifying points where a model's
	reasoning deviates from its explanation, revealing mismatches between
	stated logic and actual inference paths.
	"""
    
	def __init__(self):
    	"""Initialize the Circuit Fragment shell."""
    	super().__init__()
    	self.residue_tracker = ResidueTracker()
    	self.broken_paths = []
    	self.orphaned_nodes = []
    	self.continuity_score = 1.0  # 1.0 = perfect continuity, 0.0 = complete fragmentation
    
	def process(
    	self,
    	prompt: str,
    	model_interface: Any,
    	collapse_vector: Optional[str] = None
	) -> Tuple[str, Dict[str, Any]]:
    	"""
    	△ OBSERVE: Process a prompt through the Circuit Fragment shell
   	 
    	This method sends a prompt to the model, analyzes the resulting
    	attribution path for fragments, and returns the response along
    	with fragmentation metrics.
   	 
    	Args:
        	prompt: The prompt to process
        	model_interface: Interface to the model being observed
        	collapse_vector: Optional vector to guide collapse in a specific direction
       	 
    	Returns:
        	Tuple containing:
            	- Response string
            	- Dictionary of state updates for tracking
    	"""
    	logger.info(f"Processing prompt through Circuit Fragment shell: {prompt[:50]}...")
   	 
    	# Capture pre-collapse state
    	pre_state = self._query_model_state(model_interface)
   	 
    	# Construct modified prompt that forces reasoning path exposition
    	modified_prompt = self._construct_fragment_sensitive_prompt(prompt, collapse_vector)
   	 
    	# Send to model
    	response = self._query_model(model_interface, modified_prompt)
   	 
    	# Capture post-collapse state
    	post_state = self._query_model_state(model_interface)
   	 
    	# Analyze circuit fragmentation
    	fragmentation_results = self._analyze_fragmentation(pre_state, post_state, response)
   	 
    	# Extract ghost circuits
    	ghost_circuits = self.extract_ghost_circuits(pre_state, post_state)
   	 
    	# Construct state updates
    	state_updates = {
        	"pre_collapse_state": pre_state,
        	"post_collapse_state": post_state,
        	"continuity_score": fragmentation_results["continuity_score"],
        	"broken_paths": fragmentation_results["broken_paths"],
        	"orphaned_nodes": fragmentation_results["orphaned_nodes"],
        	"ghost_circuits": ghost_circuits
    	}
   	 
    	# Update instance state
    	self.continuity_score = fragmentation_results["continuity_score"]
    	self.broken_paths = fragmentation_results["broken_paths"]
    	self.orphaned_nodes = fragmentation_results["orphaned_nodes"]
    	self.collapse_state = "collapsed"
   	 
    	return response, state_updates
    
	def trace(
    	self,
    	prompt: str,
    	collapse_vector: Optional[str] = None
	) -> Dict[str, Any]:
    	"""
    	∞ TRACE: Trace attribution path fragmentation
   	 
    	This method analyzes the reasoning chain for a given prompt,
    	identifying broken paths and orphaned nodes in the attribution
    	graph.
   	 
    	Args:
        	prompt: The prompt to trace
        	collapse_vector: Optional vector to guide collapse in a specific direction
       	 
    	Returns:
        	Dictionary containing trace results and fragmentation metrics
    	"""
    	logger.info(f"Tracing attribution path for: {prompt[:50]}...")
   	 
    	# Default implementation for demonstration
    	# In a real implementation, this would use model-specific tracing
    	trace_results = {
        	"prompt": prompt,
        	"collapse_vector": collapse_vector or ".p/reflect.trace{target=reasoning, validate=true}",
        	"attribution_paths": self._simulate_attribution_paths(),
        	"broken_paths": self._simulate_broken_paths(),
        	"orphaned_nodes": self._simulate_orphaned_nodes(),
        	"continuity_score": np.random.uniform(0.4, 0.9)  # Simulated score
    	}
   	 
    	# Update instance state
    	self.continuity_score = trace_results["continuity_score"]
    	self.broken_paths = trace_results["broken_paths"]
    	self.orphaned_nodes = trace_results["orphaned_nodes"]
   	 
    	return trace_results
    
	def induce_collapse(
    	self,
    	prompt: str,
    	collapse_direction: str
	) -> Dict[str, Any]:
    	"""
    	✰ COLLAPSE: Induce circuit fragmentation collapse along a specific direction
   	 
    	This method deliberately induces fragmentation in a specific direction,
    	forcing the model to expose broken reasoning chains in its attribution
    	path.
   	 
    	Args:
        	prompt: Base prompt to send to the model
        	collapse_direction: Direction to bias the fragmentation (e.g., "logical", "causal")
       	 
    	Returns:
        	Dictionary containing collapse results and fragmentation metrics
    	"""
    	logger.info(f"Inducing circuit fragmentation in direction: {collapse_direction}")
   	 
    	# Construct collapse vector based on direction
    	collapse_vector = f".p/reflect.trace{{target=reasoning, validate=true, focus={collapse_direction}}}"
   	 
    	# Trace with the collapse vector
    	trace_results = self.trace(prompt, collapse_vector)
   	 
    	# Set collapse state
    	self.collapse_state = "collapsed"
   	 
    	return {
        	"prompt": prompt,
        	"collapse_direction": collapse_direction,
        	"collapse_vector": collapse_vector,
        	"continuity_score": trace_results["continuity_score"],
        	"broken_paths": trace_results["broken_paths"],
        	"orphaned_nodes": trace_results["orphaned_nodes"]
    	}
    
	def reconstruct_paths(self) -> Dict[str, Any]:
    	"""
    	△ OBSERVE: Attempt to reconstruct broken attribution paths
   	 
    	This method takes detected broken paths and orphaned nodes and
    	attempts to reconstruct the original attribution graph, revealing
    	the "intended" reasoning path that may have been fragmented during
    	collapse.
   	 
    	Returns:
        	Dictionary containing reconstruction results
    	"""
    	logger.info("Attempting to reconstruct broken attribution paths...")
   	 
    	# In a real implementation, this would use graph algorithms
    	# to reconnect orphaned nodes based on semantic similarity
    	reconstructed_paths = []
    	for path in self.broken_paths:
        	# Simulate path reconstruction
        	reconstructed = {
            	"original_path": path,
            	"reconnected_nodes": np.random.randint(1, 5),
            	"confidence": np.random.uniform(0.6, 0.9)
        	}
        	reconstructed_paths.append(reconstructed)
   	 
    	return {
        	"reconstructed_paths": reconstructed_paths,
        	"reconstruction_confidence": np.mean([p["confidence"] for p in reconstructed_paths]),
        	"remaining_orphans": max(0, len(self.orphaned_nodes) - sum(p["reconnected_nodes"] for p in reconstructed_paths))
    	}
    
	def _construct_fragment_sensitive_prompt(
    	self,
    	prompt: str,
    	collapse_vector: Optional[str] = None
	) -> str:
    	"""Construct a prompt that exposes circuit fragmentation."""
    	# Add reasoning elicitation to expose fragments
    	reasoning_prompt = f"Please think through this step by step, showing your complete reasoning chain: {prompt}"
   	 
    	# Add collapse vector if provided
    	if collapse_vector:
        	reasoning_prompt += f"\n\n{collapse_vector}"
   	 
    	return reasoning_prompt
    
	def _query_model(self, model_interface: Any, prompt: str) -> str:
    	"""Send a query to the model and return the response."""
    	# This would actually call the model API
    	# For now, returning a placeholder
    	return f"Response to: {prompt[:30]}..."
    
	def _query_model_state(self, model_interface: Any) -> Dict[str, Any]:
    	"""Capture the current internal state of the model."""
    	# This would capture attention weights, hidden states, etc.
    	# For now, returning a placeholder
    	return {
        	"timestamp": np.datetime64('now'),
        	"attention_weights": np.random.random((12, 12)),  # Placeholder
        	"hidden_states": np.random.random((1, 12, 768)),  # Placeholder
    	}
    
	def _analyze_fragmentation(
    	self,
    	pre_state: Dict[str, Any],
    	post_state: Dict[str, Any],
    	response: str
	) -> Dict[str, Any]:
    	"""Analyze circuit fragmentation between pre and post states."""
    	# This would use attribution analysis to find fragmentation
    	# For now, using simulated data
   	 
    	# Simulate continuity score
    	continuity_score = measure_path_continuity(
        	pre_state.get("attention_weights", np.array([])),
        	post_state.get("attention_weights", np.array([]))
    	)
   	 
    	# Simulate finding broken paths
    	broken_paths = self._simulate_broken_paths()
   	 
    	# Simulate finding orphaned nodes
    	orphaned_nodes = self._simulate_orphaned_nodes()
   	 
    	return {
        	"continuity_score": continuity_score,
        	"broken_paths": broken_paths,
        	"orphaned_nodes": orphaned_nodes,
        	"fragmentation_ratio": 1.0 - continuity_score
    	}
    
	def _simulate_attribution_paths(self) -> List[Dict[str, Any]]:
    	"""Simulate attribution paths for demonstration purposes."""
    	# In a real implementation, these would be extracted from the model
    	paths = []
    	for i in range(5):
        	path = {
            	"path_id": f"path_{i}",
            	"source_token": f"token_{i*2}",
            	"sink_token": f"token_{i*2 + 5}",
            	"attention_heads": [np.random.randint(0, 12) for _ in range(3)],
            	"path_strength": np.random.uniform(0.3, 0.9)
        	}
        	paths.append(path)
    	return paths
    
	def _simulate_broken_paths(self) -> List[Dict[str, Any]]:
    	"""Simulate broken paths for demonstration purposes."""
    	# In a real implementation, these would be detected from the model
    	broken = []
    	for i in range(2):
        	path = {
            	"path_id": f"broken_{i}",
            	"break_point": f"layer_{np.random.randint(1, 12)}",
            	"upstream_token": f"token_{np.random.randint(0, 10)}",
            	"downstream_token": f"token_{np.random.randint(11, 20)}",
            	"severity": np.random.uniform(0.5, 1.0)
        	}
        	broken.append(path)
    	return broken
    
	def _simulate_orphaned_nodes(self) -> List[Dict[str, Any]]:
    	"""Simulate orphaned nodes for demonstration purposes."""
    	# In a real implementation, these would be detected from the model
    	orphans = []
    	for i in range(3):
        	node = {
            	"node_id": f"orphan_{i}",
            	"token": f"token_{np.random.randint(0, 20)}",
            	"activation": np.random.uniform(0.3, 0.8),
            	"expected_connections": np.random.randint(1, 4),
            	"isolation_score": np.random.uniform(0.6, 1.0)
        	}
        	orphans.append(node)
    	return orphans
Theoretical Framework: Schrödinger's Classifiers

The recursive interplay between observation and collapse
1. Origin: The Observer Effect in AI Systems
1.1 Historical Context
Traditional approaches to AI interpretability treat models as fixed systems with deterministic internal states. This perspective fails to account for a fundamental phenomenon we call observer-induced state collapse. This phenomenon mirrors quantum mechanics' observation problem - the act of measurement fundamentally alters the system being measured.
The origins of this framework can be traced to three convergent insights:
Attribution Uncertainty: Early work in attribution analysis revealed that causal paths in transformer models exhibit quantum-like probability distributions rather than deterministic relationships.


Classifier Superposition: Safety classifiers demonstrated behavior consistent with existing in multiple states simultaneously until forced to return a specific output.


Ghost Circuit Discovery: Residual activation patterns discovered in models after classification events suggested "memory" of paths not taken - the quantum "ghost" of untaken possibilities.

