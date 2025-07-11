#################################
##### ---    IMPORTS    --- #####
#################################
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage

from src.custom_config.state_schema import CustomState
from src.custom_config.custom_messages import UserQuery, AIMessage, Action, ParameterRequestMessage, HumanMessage

from src.llm_config import *

#############################################
##### ---    CLASSES & FUNCTIONS    --- #####
#############################################

#### --- Response Classes --- #####
class OrchestratorResponse(BaseModel):
    response: str = Field(..., title="Determines which action to take next", description="Output only 'Guidelines Consultation Request', 'Clinical Case Evaluation Request', 'Finish Session Request'.")

#### ---   Sanitize Function   --- #####
def sanitize(obj):
    """
    Recursively sanitize an object by converting numpy types to native Python types,
    and ensuring that dictionaries and lists are properly formatted.
    """
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

##### ---   Process Data Patient   --- #####

def patient_processor_node(state: CustomState):
    """
    Process patient data from an Excel file, allowing the user to select which patient to process.
    Extracts data for the selected patient from all sheets, consolidating them into a nested dictionary,
    where each key is the sheet name and its value is the corresponding dictionary of parameters.
    Assigns "missing" to parameters without values.
    """
    
    print("\n#######################################################"
          "\n######## CURRENT NODE: PATIENT PROCESSOR NODE #########"
          "\n#######################################################\n"
          )

    # Step 1: Extract the Excel path from the system message
    excel_path = None
    for message in state["messages"]:
        if isinstance(message, SystemMessage) and "Excel path:" in message.content:
            excel_path = message.content.split("Excel path:")[1].strip()
            break

    if not excel_path:
        raise ValueError("Excel path not found in the system messages.")

    # Step 2: Load the Excel file
    excel_data = pd.ExcelFile(excel_path)

    # Step 3: Select the patient based on the first sheet
    first_sheet = excel_data.sheet_names[0]  # The first sheet contains patient IDs
    first_sheet_data = excel_data.parse(first_sheet)

    # Check if the sheet is empty
    if first_sheet_data.empty:
        raise ValueError(f"Sheet '{first_sheet}' is empty. No data to process.")

    # Display patient options
    print("Please, select a patient to process:")
    print("\nAvailable Patients: 1 to 20")
    # for idx, _ in first_sheet_data.iterrows():
    #     print(f"{idx + 1}")

    # Prompt the user to select a patient
    selected_row = None
    while selected_row is None:
        try:
            choice = int(input("\nEnter the number corresponding to the patient you want to process: ")) - 1
            if 0 <= choice < len(first_sheet_data):
                selected_row = first_sheet_data.iloc[choice]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Extract the patient ID for reference
    patient_id = selected_row.get("Patient ID", f"Patient {choice + 1}")

    # Step 4: Consolidate patient data from all sheets into a nested dictionary
    patient_data = {}
    for sheet in excel_data.sheet_names:
        sheet_data = excel_data.parse(sheet)
        sheet_dict = {}

        # Ensure the sheet is not empty
        if sheet_data.empty:
            print(f"Sheet '{sheet}' is empty. Marking all parameters as missing.")
            for col in sheet_data.columns:
                sheet_dict[col] = "missing"
        else:
            # Extract the row corresponding to the selected patient
            if choice < len(sheet_data):
                patient_row = sheet_data.iloc[choice]
            else:
                print(f"Sheet '{sheet}' does not contain data for Patient {choice + 1}. Marking all parameters as missing.")
                for col in sheet_data.columns:
                    sheet_dict[col] = "missing"
                patient_data[sheet] = sheet_dict
                # Continue to next sheet
                continue

            # Add data from the row
            for key, value in patient_row.items():
                if pd.isna(value):
                    value = "missing"
                sheet_dict[key] = value

        patient_data[sheet] = sanitize(sheet_dict)

    # Step 5: Update the state with consolidated patient data
    state["patient_data"] = patient_data
    state["messages"].append(AIMessage(content=f"Patient data for Patient ID {patient_id} successfully loaded."))
    print(f"\n - Patient {patient_id} loaded successfully.")
    
    print(patient_data)

    return state

##### ---   User Query Processing   --- #####

def query_input_node(state: CustomState):
    """
    Node that prompts the user to enter their query and creates a UserQuery message.
    """
    
    print("\n######################################################"
          "\n######## CURRENT NODE: QUERY INPUT NODE ##############"
          "\n######################################################\n")

    # Add a message to prompt the user to enter their query
    print("Please type your query, or write 'Goodbye' if you want to end the session:")
    state["messages"].append(AIMessage(content="Please type your query, or write 'Goodbye' if you want to end the session:"))
    state["external_messages"].append(AIMessage(content="Please type your query:"))
    # Prompt the user to enter their query
    user_input = input("Enter your query: ").strip()
    # Add the message to the state's external_messages
    state["messages"].append(UserQuery(content=user_input))
    state["external_messages"].append(UserQuery(content=user_input))

    return state


##### ---   Update Patient Data   --- #####

def patient_data_request_node(state: CustomState):
    """
    This node updates the patient data in the state prompting the user to provide values for missing parameters.
    It is invoked when the LLM agent identifies that some parameters are missing from the patient data.
    """
    # Update Current Node
    print("\n##########################################################"
          "\n######## CURRENT NODE: PATIENT DATA REQUEST NODE #########"
          "\n##########################################################\n"
        )

    last_message = state["messages"][-1]

    # Verify if the last message is a ParameterRequestMessage with user input
    if not isinstance(last_message, ParameterRequestMessage):
        raise ValueError("The last message is not a valid ParameterRequestMessage with user input.")

    # Iterate over the parameters in the ParameterRequestMessage
    for param in last_message.parameters:
        state["messages"].append(AIMessage(content=f"Please provide a value for the missing parameter: '{param}'. If you cannot provide the value, indicate 'unknown': "))
        # state["external_messages"].append(AIMessage(content=f"Please provide a value for the missing parameter: '{param}'. If you cannot provide the value, indicate 'unknown': "))
        # print(f"Please provide a value for the missing parameter: '{param}'. If you cannot provide the value, indicate 'unknown': ")
        user_value = input(f"Please provide a value for the missing parameter: '{param}'. If you cannot provide the value, indicate 'unknown': ").strip()
        state["messages"].append(HumanMessage(content=user_value))
        # state["external_messages"].append(HumanMessage(content=user_value))
        
        # If the user does not provide a value, mark it as 'unknown'
        if not user_value:
            user_value = "unknown"
        
        # Update the parameter in nested dictionaries
        for data_type in state['patient_data']:
            if param in state['patient_data'][data_type]:
                state['patient_data'][data_type][param] = user_value
                break
        
        print(f"Updated '{param}' with value: {user_value}\n")
        state["messages"].append(AIMessage(content=f"Updated '{param}' with value: {user_value}"))
        # state["external_messages"].append(AIMessage(content=f"Updated '{param}' with value: {user_value}"))


    # Return the updated state
    return state


##### ---   Orchestrator Node   --- #####

def orchestrator_node(state: CustomState):
    """
    Orchestrator node that decides the pipeline flow based on the user's query.
    Uses an LLM to interpret the user's intent and dynamically updates the plan.
    
    This node classifies the user's query into one of three categories:
    - "Guidelines Consultation Request"
    - "Clinical Case Evaluation Request"
    - "Finish Session Request"
    
    It then appends an Action message to the state with the chosen category.
    The node is designed to handle the user's query in a structured manner, ensuring that the system can respond appropriately.
    """

    print("\n###############################################"
          "\n####### CURRENT NODE: ORCHESTRATOR NODE #######"
          "\n###############################################\n"
        )

    # Get the last user query
    last_query = None
    for message in reversed(state["external_messages"]):
        if isinstance(message, UserQuery):
            last_query = message.content
            break

    if not last_query:
        raise ValueError("No user query found in the external messages.")

    print(f"### User Query Detected: {last_query} ###")

    # Ask the LLM how to classify the query
    prompt = (
        "### ROLE: ###\n"
        "You are a routing node in an interactive system that processes medical-related queries.\n"
        "Your purpose is to classify the user's query into the correct processing category and guide the flow of actions.\n\n"

        "### INSTRUCTIONS: ###\n"
        "Based on the following user query:\n"
        "{last_query}\n\n"
        
        "Classify it into one of these categories:\n\n"
    
        "- \"Guidelines Consultation Request\": Select this if the query is a simple greeting or a general information request."
        "The main intention of this category is to provide a service to search for information in official clinical guidelines about complex clinical questions. It can also apply to queries about specific patient cases if the user is seeking guidance based on established protocols.\n"
        "Also select this category if the user asks about medical procedures, treatments, or actions in hypothetical or general clinical scenarios,"
        "for example, asking how to handle a generic or fictitious patient, or what the clinical guidelines suggest in a particular case.\n"
        "Queries like 'What are the contraindications for anticoagulation?' or 'How do I manage hypotension in PE?' belong here.\n"
        "This category also applies if the user seems to be reasoning through a situation but is not referring to the patient currently selected in the system.\n\n"
    
        "- \"Clinical Case Evaluation Request\": Select this if the physician wants a comprehensive clinical evaluation and recommendations for the current selected patient.\n"
        "This triggers an automated clinical case analysis that evaluates the patient's complete clinical data, calculates risk metrics (PESI, sPESI, Risk of Early Mortality), "
        "and generates personalized diagnostic and treatment recommendations.\n"
        "Examples that trigger this evaluation:\n"
        "  • 'Evaluate my patient'\n"
        "  • 'What should I do with this patient?'\n"
        "  • 'Give me recommendations for this case'\n"
        "  • 'Analyze this patient'\n"
        "  • 'Help me with the clinical case'\n"
        "  • Any request for patient-specific clinical guidance\n\n"

        "- \"Finish Session Request\": Select this if the query indicates that the user wants to end the session. Examples include \"Goodbye\" or \"Thanks, I'm done.\"\n\n"

        "### CONSIDERATIONS: ###\n" 
        "It is crucial that you reason step-by-step before selecting the category to ensure an accurate classification.\n\n"
    
        "### OUTPUT FORMAT: ###\n"
        "Provide only the category name as the response. Strictly return one of the following: "
        "\"Guidelines Consultation Request\", \"Clinical Case Evaluation Request\", or \"Finish Session Request\"."
    )

    llm_input = prompt.format(last_query=last_query)

    llm_response = llm.with_structured_output(OrchestratorResponse).invoke(llm_input)
    print(f"### LLM Response: {llm_response.response.strip().lower()} ###")

    # Classify the query based on the LLM response
    decision = llm_response.response.strip().lower()


    # Log in internal messages for traceability
    state["messages"].append(
        SystemMessage(content=f"Planner decided next action: {decision} based on user query: \"{last_query}\".")
    )

    # Create a new Action message based on the decision
    state["messages"].append(Action(content=decision))

    print(f"### Next Action Decided: {decision} ###")

    return state


##### ---   Finish Session Node   --- #####

def finish_session_node(state: CustomState):
    """
    This node is responsible for finishing the session.
    It updates the state to indicate that the session is finished and logs a message.
    """

    print("\n########################################################"
          "\n######## CURRENT NODE: FINISH SESSION NODE #############"
          "\n########################################################\n")
    
    print("Session Finished. Thank you for using the system!")
    
    state['finish'] = True
    state["messages"].append(SystemMessage(content="Session Finished."))
    state["external_messages"].append(AIMessage(content="Session Finished."))
    
    return state
