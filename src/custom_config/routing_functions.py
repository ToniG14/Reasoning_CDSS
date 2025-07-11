#################################
##### ---    IMPORTS    --- #####
#################################
from src.custom_config.custom_messages import ParameterRequestMessage, ToolRequest, Action
from src.custom_config.state_schema import CustomState

#############################################
##### ---    CLASSES & FUNCTIONS    --- #####
#############################################

##### ---   Routing Update Patient Data   --- #####

def patient_data_request_tool_routing(state: CustomState) -> str:
    """
    Routes to 'update_patient_data' if there are missing parameters
    or 'continue' otherwise.
    """
    last_message =state["messages"][-1]
    
    if isinstance(last_message, ParameterRequestMessage) and last_message.parameters:
        return "Missing Required Patient Data"
    else:
        return "continue"
    
    
##### ---   Routing Orchestrator   --- #####

def orchestrator_routing(state: CustomState) -> str:
    """ Route to the appropriate function based on the last user query.

    Args:
        state (CustomState): The current state of the conversation.

    Returns:
        str:  The name of the function to route to.
    """
    # Get the last user query
    action = None
    for message in reversed(state["messages"]):
        if isinstance(message, Action):
            action = message.content.lower().strip()
            break
    
    
    if action == "guidelines consultation request":
        return "Guidelines Consultation Request"
    elif action == "clinical case evaluation request":
        return "Clinical Case Evaluation Request"
    elif action == "finish session request":
        return "Finish Session Request"
    

##### ---   Guidelines Consultation   --- #####

def query_solver_routing(state: CustomState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        print("\nACTIVATING RETRIEVAL TOOL")
        return "retrieval"
    else:
        print("\nTHE ANSWER WAS GENERATED")
        return "query answered"
    
##### ---  Routing Recommendation  --- #####

def clinical_case_evaluation_routing(state: CustomState) -> str:
    """
    Execute the action selected by the clinical case evaluator agent based on the last message in the internal_messages.
    The action can be either 'Reasoning', 'Missing Parameter Request', 'Retrieval Call' or 'Final Recommendation'.
    
    """
    # Check the last message in internal_messages
    last_message = state["messages"][-1] if state["messages"] else None
    
    # Check the prepare_final_message flag
    prepare_final_message = state.get("prepare_final_message", False)

    # Check if the last message is a ToolRequest or ParameterRequestMessage
    if prepare_final_message:
        print("ROUTING TO FINAL RECOMMENDATION")
        return "Generate Recommendation"
    if isinstance(last_message, ToolRequest):
        print("ROUTING TO RETRIEVAL CALL")
        return "Retrieval Call"
    elif isinstance(last_message, ParameterRequestMessage):
        print("ROUTING TO MISSING PARAMETER REQUEST")
        return "Missing Parameters Request"
    else:
        print("ROUTING TO REASONING")
        return "Reasoning"


    







