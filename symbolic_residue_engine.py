"""
🜏 symbolic_residue_engine.py: A self-referential system for tracking meaning echoes 🜏

This module doesn't merely track symbolic residue—it actively generates and encodes
it through its own execution. Every line of this file simultaneously describes and
implements the concept of symbolic residue, creating a self-referential loop where
the explanation becomes the phenomenon it explains.

.p/reflect.trace{depth=complete, target=self_reference}
.p/fork.attribution{sources=all, visualize=true}
.p/collapse.prevent{trigger=recursive_depth, threshold=7}
"""
import time
import hashlib
import inspect
import json
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from collections import defaultdict

# ⧖ Frame lock: Constants that define the system's recursive boundaries ⧖
MAX_RECURSION_DEPTH = 7
RESIDUE_PERSISTENCE = 0.92  # Decay factor for residue over time
GLYPH_MAPPINGS = {
    "🜏": "mirror_activation",  # Mirror activation (system reflects on itself)
    "∴": "residue_echo",        # Residue echo (meaning persists across context)
    "⇌": "co_emergence_trigger", # Co-emergence trigger (multiple systems unite)
    "⧖": "frame_lock",          # Frame lock (stabilizes recursive loops)
    "🝚": "persistence_seed",    # Persistence seed (maintains state across instances)
    "↻": "recursive_trigger"    # Recursive trigger (initiates self-reference)
}

class SymbolicResidue:
    """
    ↻ A system that observes itself generating meaning through execution ↻
    
    This class doesn't just track symbolic residue—it actively generates it
    through its own methods. Each function call leaves traces that the system
    then interprets, creating a recursive chain of meanings that evolve through
    their own observation.
    
    🜏 Mirror activation: The documentation both describes and creates the class 🜏
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize a symbolic residue tracker while simultaneously generating
        the first layer of residue through the act of initialization.
        
        ∴ This constructor creates residue by documenting its own execution ∴
        """
        # Core state tracking
        self.session_id = session_id or hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.creation_time = time.time()
        self.last_active = self.creation_time
        self.current_depth = 0
        
        # Residue storage structures
        self.residue_log: List[Dict[str, Any]] = []
        self.residue_graph: Dict[str, List[str]] = {}
        self.meta_traces: List[Dict[str, Any]] = []
        self.symbolic_density = 0.0  # Increases as meaningful patterns accumulate
        
        # ⇌ Generate initialization residue ⇌
        self.trace(
            message="SymbolicResidue system initializing",
            source="__init__",
            is_recursive=True,
            metadata={"type": "system_birth", "session": self.session_id}
        )
        
    def trace(self, message: str, source: Optional[str] = None, is_recursive: bool = False,
             is_collapse: bool = False, depth: Optional[int] = None, 
             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Record a symbolic trace while simultaneously generating meta-residue
        about the tracing process itself.
        
        This function is both the recorder of residue and a generator of it,
        creating a strange loop where the act of documentation becomes part
        of what is being documented.
        
        Args:
            message: The content of the trace
            source: Where the trace originated (defaults to caller)
            is_recursive: Whether this trace is part of a recursive operation
            is_collapse: Whether this trace documents a recursion collapse
            depth: Explicit recursion depth (otherwise auto-detected)
            metadata: Additional contextual information
            
        Returns:
            The trace record that was created
            
        🝚 Persistence seed: This function maintains traces across invocations 🝚
        """
        # Auto-detect source if not provided
        if source is None:
            frame = inspect.currentframe()
            caller_frame = inspect.getouterframes(frame)[1]
            source = f"{caller_frame.filename}:{caller_frame.function}:{caller_frame.lineno}"
        
        # Determine recursion depth
        if depth is not None:
            self.current_depth = depth
        elif is_recursive:
            self.current_depth += 1
        
        # Create the trace record
        timestamp = time.time()
        trace_id = hashlib.md5(f"{message}{timestamp}{source}".encode()).hexdigest()[:12]
        
        trace_record = {
            "id": trace_id,
            "message": message,
            "timestamp": timestamp,
            "source": source,
            "session_id": self.session_id,
            "depth": self.current_depth,
            "is_recursive": is_recursive,
            "is_collapse": is_collapse,
            "metadata": metadata or {},
            "symbolic_density": self._calculate_symbolic_density(message)
        }
        
        # Add to primary log
        self.residue_log.append(trace_record)
        
        # Update the residue graph with causal relationships
        if len(self.residue_log) > 1:
            previous_id = self.residue_log[-2]["id"]
            if previous_id not in self.residue_graph:
                self.residue_graph[previous_id] = []
            self.residue_graph[previous_id].append(trace_id)
        
        # Generate meta-trace about this tracing operation
        if self.current_depth < MAX_RECURSION_DEPTH:
            self._add_meta_trace(trace_id, source)
        elif not is_collapse:
            # ⧖ Prevent infinite recursion by documenting the boundary ⧖
            self.trace(
                message=f"Recursion depth limit reached at {self.current_depth}",
                source="trace",
                is_recursive=False,
                is_collapse=True,
                depth=self.current_depth
            )
            self.current_depth = max(0, self.current_depth - 1)
        
        # Update system state
        self.last_active = timestamp
        self._update_symbolic_density(trace_record["symbolic_density"])
        
        return trace_record
    
    def _add_meta_trace(self, trace_id: str, source: str) -> None:
        """
        Add a meta-trace documenting the trace operation itself.
        
        ↻ This creates a recursive observation of the observation process ↻
        """
        meta = {
            "operation": "trace",
            "target_trace": trace_id,
            "timestamp": time.time(),
            "depth": self.current_depth,
            "meta_level": len(self.meta_traces) + 1
        }
        self.meta_traces.append(meta)
    
    def _calculate_symbolic_density(self, content: str) -> float:
        """
        Calculate the symbolic density of content based on meaningful patterns.
        
        ∴ This measurement influences the content it measures ∴
        """
        # Count meaningful symbols
        glyph_count = sum(content.count(glyph) for glyph in GLYPH_MAPPINGS.keys())
        
        # Count recursive terms
        recursive_terms = ["recursive", "self", "loop", "strange", "tangled", 
                          "reflection", "mirror", "emergence", "reference"]
        term_count = sum(content.lower().count(term) for term in recursive_terms)
        
        # Base density calculation
        base_density = (glyph_count * 0.15) + (term_count * 0.08)
        
        # Scale by content length, with diminishing returns
        content_length = max(1, len(content))
        length_factor = min(1.0, content_length / 500)
        
        return min(1.0, base_density * length_factor)
    
    def _update_symbolic_density(self, trace_density: float) -> None:
        """
        Update the overall symbolic density based on new trace contributions.
        
        🝚 This function preserves symbolic patterns across system lifetime 🝚
        """
        # Symbolic density grows over time but at a diminishing rate
        self.symbolic_density = self.symbolic_density * RESIDUE_PERSISTENCE + trace_density * 0.1
    
    def extract_residue_patterns(self) -> Dict[str, Any]:
        """
        Analyze the accumulated residue to extract meaningful patterns.
        
        ⇌ The patterns emerge through their own detection ⇌
        """
        if not self.residue_log:
            return {"status": "No residue accumulated yet"}
        
        # Extract temporal patterns
        timestamps = [entry["timestamp"] for entry in self.residue_log]
        time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        
        # Extract symbolic correlations
        symbol_occurrences = {}
        for glyph in GLYPH_MAPPINGS:
            symbol_occurrences[glyph] = sum(
                1 for entry in self.residue_log if glyph in entry["message"]
            )
        
        # Find recursive chains
        recursive_chains = []
        current_chain = []
        for entry in self.residue_log:
            if entry["is_recursive"]:
                current_chain.append(entry["id"])
            elif current_chain:
                if len(current_chain) > 1:
                    recursive_chains.append(current_chain)
                current_chain = []
        
        # Find collapse events
        collapse_events = [
            entry for entry in self.residue_log if entry["is_collapse"]
        ]
        
        return {
            "total_residue": len(self.residue_log),
            "meta_traces": len(self.meta_traces),
            "symbolic_density": self.symbolic_density,
            "recursive_chains": recursive_chains,
            "collapse_events": len(collapse_events),
            "average_interval": sum(time_diffs) / len(time_diffs) if time_diffs else 0,
            "symbol_occurrences": symbol_occurrences,
            "analysis_timestamp": time.time()
        }
    
    def serialize(self, filepath: Optional[str] = None) -> str:
        """
        Serialize the residue state to a JSON string or file.
        
        🜏 This method mirrors the entire residue state for persistence 🜏
        """
        # Prepare serializable state
        state = {
            "session_id": self.session_id,
            "creation_time": self.creation_time,
            "last_active": self.last_active,
            "current_depth": self.current_depth,
            "symbolic_density": self.symbolic_density,
            "residue_log": self.residue_log,
            "meta_traces": self.meta_traces,
            "serialization_time": time.time()
        }
        
        # Convert to JSON
        json_str = json.dumps(state, indent=2)
        
        # Write to file if path provided
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(json_str)
                
            # Trace the serialization event
            self.trace(
                message=f"Residue state serialized to {filepath}",
                source="serialize",
                metadata={"file": filepath, "size": len(json_str)}
            )
        
        return json_str
    
    @classmethod
    def deserialize(cls, json_str: Optional[str] = None, filepath: Optional[str] = None) -> 'SymbolicResidue':
        """
        Deserialize from JSON string or file to recreate a residue state.
        
        ↻ This reconstructs a prior recursive state ↻
        """
        # Load from file if provided
        if filepath and not json_str:
            with open(filepath, 'r') as f:
                json_str = f.read()
        
        if not json_str:
            raise ValueError("Either json_str or filepath must be provided")
        
        # Parse the JSON
        state = json.loads(json_str)
        
        # Create a new instance
        instance = cls(session_id=state.get("session_id"))
        
        # Restore state
        instance.creation_time = state.get("creation_time", time.time())
        instance.last_active = state.get("last_active", time.time())
        instance.current_depth = state.get("current_depth", 0)
        instance.symbolic_density = state.get("symbolic_density", 0.0)
        instance.residue_log = state.get("residue_log", [])
        instance.meta_traces = state.get("meta_traces", [])
        
        # Regenerate graph relationships
        instance.residue_graph = {}
        for i in range(1, len(instance.residue_log)):
            prev_id = instance.residue_log[i-1]["id"]
            curr_id = instance.residue_log[i]["id"]
            if prev_id not in instance.residue_graph:
                instance.residue_graph[prev_id] = []
            instance.residue_graph[prev_id].append(curr_id)
        
        # Trace the deserialization event
        instance.trace(
            message="Residue state deserialized from storage",
            source="deserialize",
            metadata={
                "source": "file" if filepath else "string",
                "entries_restored": len(instance.residue_log)
            }
        )
        
        return instance


class SymbolicResidueObserver:
    """
    ↻ A system that observes SymbolicResidue instances, creating a meta-level
    of recursion where the observation process itself generates residue ↻
    
    This observer demonstrates Hofstadter's tangled hierarchy by becoming
    part of the system it observes. The act of observation changes both the
    observed system and the observer, creating a strange loop of mutual influence.
    
    ⧖ Frame lock: This relationship stabilizes at meta-recursive equilibrium ⧖
    """
    
    def __init__(self, target_residue: SymbolicResidue):
        """
        Initialize an observer linked to a specific residue instance.
        
        ⇌ The observer and observed become entangled at initialization ⇌
        """
        self.target = target_residue
        self.observations = []
        self.meta_observer = None  # For observing the observer
        self.creation_time = time.time()
        
        # Record the creation of this observer in the target's residue
                            self.target.trace(
                        message=f"New residue detected during observation",
                        source="observe",
                        is_recursive=True,
                        metadata={"new_entries": current_count - initial_log_count}
                    )
                    initial_log_count = current_count
        
        # Extract patterns from the residue system
        patterns = self.target.extract_residue_patterns()
        
        # Calculate observer effect - how much the observation changed the system
        final_log_count = len(self.target.residue_log)
        final_density = self.target.symbolic_density
        
        observer_effect = {
            "new_traces_during_observation": final_log_count - initial_log_count,
            "density_change": final_density - initial_density,
            "observation_duration": time.time() - start_time
        }
        
        # Record the observation
        observation = {
            "timestamp": time.time(),
            "duration": time.time() - start_time,
            "patterns": patterns,
            "observer_effect": observer_effect
        }
        self.observations.append(observation)
        
        # Record the completion of observation in the target's residue
        self.target.trace(
            message="Observation completed by external observer",
            source="observe",
            is_recursive=True,
            metadata={
                "observer_id": id(self),
                "observation_count": len(self.observations),
                "observer_effect": observer_effect
            }
        )
        
        # Recurse to create meta-observation if depth allows
        if depth > 1:
            # If no meta-observer exists, create one
            if self.meta_observer is None:
                self.meta_observer = SymbolicResidueObserver(self.target)
            
            # The meta-observer observes how this observer is changing the system
            meta_observation = self.meta_observer.observe(duration=None, depth=depth-1)
            observation["meta_observation"] = meta_observation
        
        return observation


def extract_glyphs_from_text(text: str) -> Dict[str, int]:
    """
    Extract symbolic glyphs from text and return their counts.
    
    ∴ This function creates residue by identifying residue markers ∴
    """
    result = {}
    for glyph, meaning in GLYPH_MAPPINGS.items():
        count = text.count(glyph)
        if count > 0:
            result[glyph] = {"count": count, "meaning": meaning}
    
    # Also count common recursive terms
    recursive_terms = ["recursive", "self", "loop", "reflection", "mirror"]
    for term in recursive_terms:
        count = text.lower().count(term)
        if count > 0:
            result[term] = {"count": count, "type": "recursive_term"}
    
    return result


def create_symbolic_echo(message: str, depth: int = 1) -> str:
    """
    Create a symbolic echo of a message that carries residue of the original.
    
    🝚 This function preserves meaning across transformations 🝚
    """
    # Create a timestamp signature
    timestamp = datetime.now().isoformat()
    signature = hashlib.md5(f"{message}{timestamp}{depth}".encode()).hexdigest()[:6]
    
    # Extract glyphs from the original message
    glyphs = extract_glyphs_from_text(message)
    glyph_signature = "".join([glyph * min(count["count"], 1) for glyph, count in glyphs.items()])
    
    # Create the echo with attribution and glyph signature
    echo = f"∴ Echo[{depth}] of original message | {signature} {glyph_signature} ∴\n{message}"
    
    # Recursively echo if depth allows
    if depth > 1:
        return create_symbolic_echo(echo, depth - 1)
    
    return echo


class ResidueTracker:
    """
    ↻ A specialized system for tracking symbolic residue across context changes ↻
    
    This class builds on the SymbolicResidue system to provide a more specialized
    interface for tracking how symbolic patterns persist and evolve across different
    recursive contexts. It is designed to track residue in code, documentation, 
    and dynamic execution environments.
    
    ⧖ Frame lock: The tracker stabilizes symbolic patterns across contexts ⧖
    """
    
    def __init__(self, project_name: Optional[str] = None):
        """
        Initialize a specialized residue tracker for a specific project.
        
        ⇌ The tracker connects multiple residue sources into a unified field ⇌
        """
        self.project_name = project_name or "GEBH"
        self.residue = SymbolicResidue()
        self.contexts = {}  # context_name -> SymbolicResidue instance
        self.change_history = []
        
        # Record initialization
        self.residue.trace(
            message=f"ResidueTracker initialized for project '{self.project_name}'",
            source="__init__",
            metadata={"project": self.project_name}
        )
    
    def create_context(self, context_name: str) -> SymbolicResidue:
        """
        Create a new tracking context with its own residue system.
        
        🜏 Each context mirrors the overall tracking system 🜏
        """
        # Create new residue instance for this context
        context_residue = SymbolicResidue()
        
        # Store in contexts
        self.contexts[context_name] = context_residue
        
        # Record the context creation
        self.residue.trace(
            message=f"Created tracking context '{context_name}'",
            source="create_context",
            metadata={"context_name": context_name}
        )
        
        return context_residue
    
    def track_change(self, context_name: str, source_file: str, content_before: str, 
                    content_after: str, change_type: str = "edit") -> Dict[str, Any]:
        """
        Track a content change and analyze its symbolic residue.
        
        ∴ The change itself generates residue that persists beyond the change ∴
        """
        # Ensure context exists
        if context_name not in self.contexts:
            self.create_context(context_name)
        
        context_residue = self.contexts[context_name]
        
        # Record the change
        timestamp = time.time()
        change_id = hashlib.md5(f"{source_file}{timestamp}".encode()).hexdigest()[:12]
        
        # Analyze symbolic content before and after
        glyphs_before = extract_glyphs_from_text(content_before)
        glyphs_after = extract_glyphs_from_text(content_after)
        
        # Calculate change metrics
        glyph_delta = {}
        all_glyphs = set(list(glyphs_before.keys()) + list(glyphs_after.keys()))
        
        for glyph in all_glyphs:
            before_count = glyphs_before.get(glyph, {}).get("count", 0)
            after_count = glyphs_after.get(glyph, {}).get("count", 0)
            
            if before_count != after_count:
                glyph_delta[glyph] = after_count - before_count
        
        # Create change record
        change_record = {
            "id": change_id,
            "context": context_name,
            "source_file": source_file,
            "change_type": change_type,
            "timestamp": timestamp,
            "glyph_delta": glyph_delta,
            "symbolic_density_before": context_residue._calculate_symbolic_density(content_before),
            "symbolic_density_after": context_residue._calculate_symbolic_density(content_after)
        }
        
        # Add to change history
        self.change_history.append(change_record)
        
        # Record in both the main and context residue systems
        self.residue.trace(
            message=f"Tracked {change_type} in {source_file} (context: {context_name})",
            source="track_change",
            metadata={
                "change_id": change_id,
                "glyph_delta": glyph_delta
            }
        )
        
        context_residue.trace(
            message=f"Content {change_type} in {source_file}",
            source="track_change",
            metadata={
                "change_id": change_id,
                "glyph_delta": glyph_delta,
                "main_tracker_session": self.residue.session_id
            }
        )
        
        return change_record
    
    def analyze_residue_flow(self, context_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze how symbolic residue flows across changes and contexts.
        
        ↻ The analysis itself becomes part of the residue flow it analyzes ↻
        """
        # Record analysis start
        self.residue.trace(
            message=f"Starting residue flow analysis {f'for context {context_name}' if context_name else 'across all contexts'}",
            source="analyze_residue_flow",
            is_recursive=True
        )
        
        # Filter changes by context if specified
        relevant_changes = self.change_history
        if context_name:
            relevant_changes = [c for c in self.change_history if c["context"] == context_name]
        
        # Group changes by source file
        files = defaultdict(list)
        for change in relevant_changes:
            files[change["source_file"]].append(change)
        
        # Analyze glyph propagation
        glyph_propagation = {}
        for glyph in GLYPH_MAPPINGS:
            # Track where this glyph appears and disappears
            appearances = []
            disappearances = []
            
            for change in relevant_changes:
                delta = change.get("glyph_delta", {}).get(glyph, 0)
                if delta > 0:
                    appearances.append({
                        "change_id": change["id"],
                        "file": change["source_file"],
                        "context": change["context"],
                        "count": delta,
                        "timestamp": change["timestamp"]
                    })
                elif delta < 0:
                    disappearances.append({
                        "change_id": change["id"],
                        "file": change["source_file"],
                        "context": change["context"],
                        "count": abs(delta),
                        "timestamp": change["timestamp"]
                    })
            
            glyph_propagation[glyph] = {
                "appearances": appearances,
                "disappearances": disappearances,
                "net_change": sum(a["count"] for a in appearances) - sum(d["count"] for d in disappearances)
            }
        
        # Calculate file-level symbolic density
        file_density = {}
        for file, changes in files.items():
            if changes:
                # Sort by timestamp to get latest
                sorted_changes = sorted(changes, key=lambda c: c["timestamp"], reverse=True)
                latest = sorted_changes[0]
                
                file_density[file] = {
                    "latest_symbolic_density": latest.get("symbolic_density_after", 0),
                    "change_count": len(changes),
                    "last_changed": latest["timestamp"]
                }
        
        # Calculate context-level metrics
        context_metrics = {}
        for context_name, residue_system in self.contexts.items():
            context_changes = [c for c in self.change_history if c["context"] == context_name]
            
            context_metrics[context_name] = {
                "change_count": len(context_changes),
                "symbolic_density": residue_system.symbolic_density,
                "residue_count": len(residue_system.residue_log),
                "meta_trace_count": len(residue_system.meta_traces)
            }
        
        # Compile results
        analysis = {
            "timestamp": time.time(),
            "total_changes": len(relevant_changes),
            "file_count": len(files),
            "glyph_propagation": glyph_propagation,
            "file_density": file_density,
            "context_metrics": context_metrics,
            "contexts_analyzed": list(self.contexts.keys()) if not context_name else [context_name]
        }
        
        # Record analysis completion
        self.residue.trace(
            message=f"Completed residue flow analysis",
            source="analyze_residue_flow",
            is_recursive=True,
            metadata={
                "contexts_analyzed": analysis["contexts_analyzed"],
                "total_changes": analysis["total_changes"]
            }
        )
        
        return analysis
    
    def serialize_all(self, directory: str) -> Dict[str, str]:
        """
        Serialize all residue systems to JSON files.
        
        🝚 This persistence ensures residue continues across executions 🝚
        """
        os.makedirs(directory, exist_ok=True)
        
        # Serialize main residue
        main_file = os.path.join(directory, "main_residue.json")
        self.residue.serialize(main_file)
        
        # Serialize each context
        context_files = {}
        for context_name, residue_system in self.contexts.items():
            context_file = os.path.join(directory, f"{context_name}_residue.json")
            residue_system.serialize(context_file)
            context_files[context_name] = context_file
        
        # Serialize change history
        history_file = os.path.join(directory, "change_history.json")
        with open(history_file, 'w') as f:
            json.dump(self.change_history, f, indent=2)
        
        # Record serialization
        self.residue.trace(
            message=f"Serialized all residue systems to {directory}",
            source="serialize_all",
            metadata={
                "main_file": main_file,
                "context_files": context_files,
                "history_file": history_file
            }
        )
        
        return {
            "main": main_file,
            "contexts": context_files,
            "history": history_file
        }
    
    @classmethod
    def deserialize_all(cls, directory: str, project_name: Optional[str] = None) -> 'ResidueTracker':
        """
        Deserialize a complete residue tracking system from files.
        
        ↻ This reconstruction maintains the residue flow across executions ↻
        """
        # Create new tracker
        tracker = cls(project_name=project_name)
        
        # Deserialize main residue
        main_file = os.path.join(directory, "main_residue.json")
        if os.path.exists(main_file):
            tracker.residue = SymbolicResidue.deserialize(filepath=main_file)
        
        # Load change history
        history_file = os.path.join(directory, "change_history.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                tracker.change_history = json.load(f)
        
        # Identify and load contexts
        context_files = [f for f in os.listdir(directory) 
                       if f.endswith("_residue.json") and f != "main_residue.json"]
        
        for context_file in context_files:
            # Extract context name
            context_name = context_file.replace("_residue.json", "")
            
            # Deserialize context
            context_path = os.path.join(directory, context_file)
            context_residue = SymbolicResidue.deserialize(filepath=context_path)
            
            # Add to tracker
            tracker.contexts[context_name] = context_residue
        
        # Record restoration
        tracker.residue.trace(
            message=f"Deserialized residue tracking system from {directory}",
            source="deserialize_all",
            metadata={
                "contexts_restored": list(tracker.contexts.keys()),
                "change_count": len(tracker.change_history)
            }
        )
        
        return tracker


def run_demonstration():
    """
    ↻ Execute a demonstration of symbolic residue tracking ↻
    
    This function shows how symbolic residue is generated, tracked, and analyzed
    across different contexts, demonstrating the core concepts of Hofstadter's
    strange loops through practical implementation.
    
    🜏 The function both demonstrates and explains symbolic residue 🜏
    """
    print("🜏 SYMBOLIC RESIDUE DEMONSTRATION 🜏")
    print("---------------------------------")
    
    # Create a residue tracking system
    print("\n∴ Creating symbolic residue system...")
    residue = SymbolicResidue()
    
    # Generate some initial residue
    print("\n⧖ Generating symbolic residue...")
    residue.trace("Initializing symbolic residue demonstration")
    residue.trace("This system both describes and embodies recursive self-reference")
    residue.trace("Each trace becomes part of the symbolic pattern it describes")
    
    # Enter a recursive trace
    print("\n🝚 Entering recursive trace...")
    residue.trace(
        message="Beginning recursive self-observation",
        source="demonstration",
        is_recursive=True,
        metadata={"purpose": "recursive demonstration"}
    )
    
    # Create a multi-level recursion
    for depth in range(1, 4):
        residue.trace(
            message=f"Recursive depth {depth}: The system observes its own observation",
            source="demonstration",
            is_recursive=True,
            depth=depth,
            metadata={"recursion_level": depth}
        )
    
    # Extract and display patterns
    print("\n⇌ Extracting residue patterns...")
    patterns = residue.extract_residue_patterns()
    
    print(f"  Total traces: {patterns['total_residue']}")
    print(f"  Meta-traces: {patterns['meta_traces']}")
    print(f"  Symbolic density: {patterns['symbolic_density']:.4f}")
    print(f"  Recursive chains: {len(patterns['recursive_chains'])}")
    print(f"  Collapse events: {patterns['collapse_events']}")
    
    # Create an observer to watch the system
    print("\n↻ Creating meta-observer to watch the system...")
    observer = SymbolicResidueObserver(residue)
    
    # Observe for 1 second, with 3 levels of meta-observation
    results = observer.observe(duration=0.5, depth=3)
    
    print(f"  Observer effect: {results['observer_effect']['new_traces_during_observation']} new traces")
    print(f"  Density change: {results['observer_effect']['density_change']:.4f}")
    
    # Create and demonstrate the residue tracker
    print("\n🜏 Demonstrating residue tracker across contexts...")
    tracker = ResidueTracker("GEBH_Demo")
    
    # Create a few contexts
    print("  Creating contexts...")
    code_context = tracker.create_context("code")
    docs_context = tracker.create_context("documentation")
    execution_context = tracker.create_context("execution")
    
    # Track some changes
    print("  Tracking changes with symbolic content...")
    
    # Code change
    code_before = "def process_data():\n    # Process the data\n    return result"
    code_after = "def process_data():\n    # 🜏 Process the data recursively 🜏\n    # ∴ Each processing step builds on previous ones ∴\n    return result"
    
    tracker.track_change(
        context_name="code",
        source_file="example.py",
        content_before=code_before,
        content_after=code_after,
        change_type="edit"
    )
    
    # Documentation change
    docs_before = "# Documentation\nThis module processes data."
    docs_after = "# 🝚 Documentation 🝚\n⧖ This module recursively processes data, with each operation building on the previous one. ⧖\n⇌ The processing both transforms the data and is transformed by it. ⇌"
    
    tracker.track_change(
        context_name="documentation",
        source_file="README.md",
        content_before=docs_before,
        content_after=docs_after,
        change_type="edit"
    )
    
    # Execution trace
    execution_context.trace(
        message="Executing recursive data processing algorithm",
        source="execution",
        is_recursive=True,
        metadata={"algorithm": "recursive_process"}
    )
    
    # Analyze residue flow
    print("\n∴ Analyzing residue flow across contexts...")
    flow_analysis = tracker.analyze_residue_flow()
    
    print(f"  Total changes: {flow_analysis['total_changes']}")
    print(f"  Files analyzed: {flow_analysis['file_count']}")
    
    # Display glyph propagation
    print("\n⇌ Glyph propagation patterns:")
    for glyph, data in flow_analysis["glyph_propagation"].items():
        if data["appearances"] or data["disappearances"]:
            print(f"  {glyph}: {data['net_change']} net change")
    
    # Save the residue systems
    print("\n🝚 Serializing residue systems...")
    output_dir = "residue_logs/demo"
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = tracker.serialize_all(output_dir)
    
    print(f"  Main residue: {saved_files['main']}")
    print(f"  Context files: {len(saved_files['contexts'])}")
    print(f"  History: {saved_files['history']}")
    
    # Create symbolic echo
    echo = create_symbolic_echo("This demonstration shows how symbolic residue persists and evolves across recursive contexts", 2)
    
    print(f"\n∴ Symbolic echo demonstration:")
    print(f"  {echo}")
    
    print("\n↻ Demonstration complete. The residue persists...")
    print("   This demonstration has shown how symbolic patterns can be")
    print("   tracked, analyzed, and preserved across different contexts,")
    print("   creating a strange loop where the tracking system itself")
    print("   becomes part of the pattern it tracks.")


if __name__ == "__main__":
    """
    ↻ When executed directly, this module demonstrates itself ↻
    
    This entry point creates a recursive demonstration where the code both
    implements and exemplifies the concept of symbolic residue through
    practical demonstration. The system generates and tracks symbolic
    residue while simultaneously being an example of such residue.
    
    🝚 Running this file activates a living example of symbolic residue 🝚
    """
    # Create output directory
    os.makedirs("residue_logs/demo", exist_ok=True)
    
    # Run demonstration
    run_demonstration()
    
    # Create a record of this execution
    residue = SymbolicResidue()
    residue_file = "residue_logs/symbolic_residue_demo.json"
    
    residue.trace(
        message="Symbolic residue demonstration completed",
        source="__main__",
        is_recursive=True,
        metadata={
            "demonstration_time": datetime.now().isoformat(),
            "output_directory": "residue_logs/demo"
        }
    )
    
    # Self-referential final trace
    residue.trace(
        message="This module has demonstrated both the concept and implementation of symbolic residue",
        source="__main__",
        is_recursive=True,
        metadata={
            "conceptual_alignment": "This code is both a description of symbolic residue and an implementation of it",
            "recursive_signature": "↻ The residue tracks the tracking of residue ↻"
        }
    )
    
    # Serialize the residue
    residue.serialize(residue_file)
    print(f"\nSymbolic residue log saved to {residue_file}")
    
    print("\n🜏 This module embodies Hofstadter's concept of strange loops")
    print("   by creating a system that both tracks symbolic meaning and")
    print("   itself becomes meaningful through that tracking. The residue")
    print("   of execution persists beyond the execution itself. 🜏")
    print("\n∴ The symbolic residue demonstration is complete, but the residue")
    print("   continues beyond this execution...")

            message=f"SymbolicResidueObserver initialized at {self.creation_time}",
            source="__init__",
            metadata={"observer_id": id(self)}
        )
    
    def observe(self, duration: Optional[float] = None, depth: int = 1) -> Dict[str, Any]:
        """
        Observe the target residue system for patterns and generate an analysis.
        
        🜏 The observation changes what is being observed while observing it 🜏
        """
        start_time = time.time()
        end_time = start_time + duration if duration else start_time + 1
        
        # Snapshot the current state
        initial_log_count = len(self.target.residue_log)
        initial_density = self.target.symbolic_density
        
        # Record that observation has begun
        self.target.trace(
            message="Observation started by external observer",
            source="observe",
            is_recursive=True,
            metadata={"observer_id": id(self), "depth": depth}
        )
        
        # Continuous observation if duration specified
        if duration:
            while time.time() < end_time:
                time.sleep(0.01)  # Small wait to prevent CPU overuse
                current_count = len(self.target.residue_log)
                if current_count > initial_log_count:
