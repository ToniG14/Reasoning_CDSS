#################################
##### ---    IMPORTS    --- #####
#################################
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START

# Import the CustomState schema
from src.custom_config.state_schema import CustomState

# Import the nodes from the different subgraphs
from src.nodes.common_nodes import *
from src.nodes.metrics_calculation_nodes import *
from src.nodes.guidelines_consultation_nodes import *
from src.nodes.clinical_case_evaluation_nodes import *

# Import the routing functions
from src.custom_config.routing_functions import *

##########################################################
##### ---    GRAPH DEFINITION AND COMPILATION    --- #####
##########################################################

##### ---    GRAPH DEFINITION    --- #####
def compile_graph() -> StateGraph:
    """
    Compiles the graph with the defined nodes and edges.

    Returns:
        - StateGraph: The compiled graph with the defined nodes and edges.
    """
    ### Initialize the graph with CustomState
    graph = StateGraph(CustomState)

    ### Add initial nodes
    graph.add_node("Patient Processor", patient_processor_node)
    graph.add_node("Query Input", query_input_node)
    graph.add_node("Orchestrator", orchestrator_node)

    ### Add the GUIDELINES CONSULTATION subgraph node to the graph
    graph.add_node("Query Solver",query_solver_node)
    graph.add_node("Dynamic Retrieval Tool",retrieval_tool_node)

    ### Add the CLINICAL CASE EVALUATION subgraph to the graph
    # PESI and sPESI
    graph.add_node("PESI Parameters Evaluator", pesi_parameters_evaluator_node)
    graph.add_node("PESI Patient Data Request Tool", patient_data_request_node) # For missing data in PESI calculation
    graph.add_node("PESI Calculator", pesi_calculator_node)

    # Risk Of Early Mortality calculation nodes
    graph.add_node("ROEM Parameters Evaluator", roem_parameters_evaluator_node)
    graph.add_node("ROEM Patient Data Request Tool", patient_data_request_node) # For missing data in PESI (BOTH) calculation
    graph.add_node("ROEM Calculator", roem_calculator_node)

    # Evaluator Agents
    graph.add_node("Clinical Case Evaluator", clinical_case_evaluator_node)
    graph.add_node("Patient Data Request Tool", patient_data_request_node)  # For missing data in recommendations
    graph.add_node("Dynamic Retrieval Tool 2", retrieval_tool_node_2)
    graph.add_node("Clinical Case Report Generator", clinical_case_report_generator_node)

    ###  Finish Session Node
    graph.add_node("Finish Session", finish_session_node)

    ##### ---    GRAPH COMPILATION    --- #####

    # Define edges
    # Step 1: Start with patient data processing and getting the user query. Then, route to the orchestrator
    graph.add_edge(START, "Patient Processor")
    graph.add_edge("Patient Processor", "Query Input")
    graph.add_edge("Query Input", "Orchestrator")

    # Step 2: Conditional routing based on the orchestrator's decision
    graph.add_conditional_edges(
        source="Orchestrator",
        path=orchestrator_routing,
        path_map={
            "Guidelines Consultation Request": "Query Solver",
            "Clinical Case Evaluation Request": "PESI Parameters Evaluator",
            "Finish Session Request": "Finish Session"
        }
    )

    # Step 3.1: Process the Guidelines Consultation subgraph
    
    graph.add_conditional_edges(
        source="Query Solver",
        path=query_solver_routing,
        path_map={
            "retrieval": "Dynamic Retrieval Tool",
            "query answered": "Query Input"
        }
    )

    graph.add_edge("Dynamic Retrieval Tool", "Query Solver")

    # Step 3.2: Handle the Clinical Case Evaluation subgraph
    
    # Obtain PESI
    graph.add_conditional_edges(
        source="PESI Parameters Evaluator",
        path=patient_data_request_tool_routing,
        path_map={
            "Missing Required Patient Data": "PESI Patient Data Request Tool",
            "continue": "PESI Calculator"
        }
    )

    graph.add_edge("PESI Patient Data Request Tool", "PESI Calculator")
    graph.add_edge("PESI Calculator", "ROEM Parameters Evaluator")

    # Obtain ROEM
    graph.add_conditional_edges(
        source="ROEM Parameters Evaluator",
        path=patient_data_request_tool_routing,
        path_map={
            "Missing Required Patient Data": "ROEM Patient Data Request Tool",
            "continue": "ROEM Calculator"
        }
    )

    graph.add_edge("ROEM Patient Data Request Tool", "ROEM Calculator")
    graph.add_edge("ROEM Calculator", "Clinical Case Evaluator")

    # Clinical Case Evaluator Routing
    graph.add_conditional_edges(
        source="Clinical Case Evaluator",
        path=clinical_case_evaluation_routing,
        path_map={
            "Generate Recommendation": "Clinical Case Report Generator",
            "Retrieval Call": "Dynamic Retrieval Tool 2",
            "Missing Parameters Request": "Patient Data Request Tool",
            "Reasoning": "Clinical Case Evaluator",
        }
    )

    graph.add_edge("Dynamic Retrieval Tool 2", "Clinical Case Evaluator")
    graph.add_edge("Patient Data Request Tool", "Clinical Case Evaluator")
    graph.add_edge("Clinical Case Report Generator", "Query Input")

    # Compile the graph with memory saving
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)

    return compiled_graph

#### ---    GRAPH VISUALIZATION    --- ####
def visualize_graph(graph: StateGraph, save_name: str = None):
    """
    Visualize the computation graph and optionally save it as an image.
    
    Parameters:
        - graph (StateGraph): The graph to visualize.
        - save_name (str): The name to save the graph image. If None, the image
        will not be saved.
    """
    if save_name:
        image = graph.get_graph(xray=1).draw_mermaid_png()
        with open(f"{save_name}.png", "wb") as f:
            f.write(image)