#################################
##### ---    IMPORTS    --- #####
#################################
from pydantic import BaseModel, Field
from typing import List
from src.services.retrieval import retrieve
from src.custom_config.state_schema import CustomState
from src.custom_config.custom_messages import *
from langchain_core.messages import SystemMessage

from src.llm_config import *

#############################################
##### ---    CLASSES & FUNCTIONS    --- #####
#############################################

##### --- Custom Responses --- #####

class RequiredParamsResponse(BaseModel):
    required_params: List[str] = Field(..., title="Required Parameters to calculate a given metric.")

class MissingParametersResponse(BaseModel):
    missing_params: List[str] = Field(..., title="Missing Parameters to calculate a given metric.")

##### ---   Identify PESI Parameters   --- #####

def pesi_parameters_evaluator_node(state: CustomState):
    """
    This node identifies the required parameters for the calculation of PESI and sPESI scores using an LLM.
    It checks for any missing data in state["patient_data"] and can invoke the Patient Data Request Tool if necessary to request missing parameters.
    
    The node is designed to ensure that all necessary parameters are available before proceeding with the PESI and sPESI calculations.
    """
    
    print("\n#############################################################"
          "\n######## CURRENT NODE: PESI & sPESI CALCULATOR NODE #########"
          "\n#############################################################\n"
        )
    
    # FIRST: CHECK IF PESI HAS ALREADY BEEN CALCULATED
    if "PESI_Value" in state["metrics"]:
        print("PESI score has already been calculated.")
        state["messages"].append(SystemMessage(content="PESI and sPESI score has already been calculated."))
        return state

    # Step 1: Retrieve PESI calculation guidelines
    retrieval_query = "Parameters required to calculate the PESI score and sPESI score"
    docs_content , documents = retrieve(query = retrieval_query, top_k = 1, reranking= False, include_refs = False)
    
    # Step 2: Use LLM to extract required parameters
    prompt = (
        "### ROLE ###\n"
        "You are a medical expert specialized in the PESI score calculation.\n\n"
        
        "### INSTRUCTIONS ###\n"
        "You are given a set of guidelines for calculating the PESI score.\n"
        "Your task is to extract the parameters needed to calculate the PESI score.\n"
        "Base the response on the following guidelines:\n"
        "{docs_content}\n\n"
        
        "### CONSIDERATIONS: ###\n"
        "Please, it is CRUCIAL for your task that you consider \"male sex\" as \"sex\".\n"
        "If you found \"-\" in the information of the guidelines, it means that that parameter is not needed to calculate the corresponding metric.\n\n"

        
        "### OUTPUT FORMAT ###\n"
        "Your response must be strictly a list with the needed data parameters to calculate the given metrics.\n"
        "Use STRICTLY this format:\n"
        "Response: param1, param2, param3, ...\n\n"
    )
    
    llm_input_req_params = prompt.format(docs_content=docs_content)
    
    llm_pesi_params = llm.with_structured_output(RequiredParamsResponse).invoke(llm_input_req_params)

    print(f"###### LLM response: #### \n\n {llm_pesi_params.required_params}")

    # Parse LLM response to extract required parameters
    pesi_params = llm_pesi_params.required_params
    required_parameters = {"PESI": pesi_params}

    # print(f"###### Parsed Parameters: #### \n\n {required_parameters}")
    
    # Step 3: Identify missing parameters
    prompt = (
        "### ROLE: ###\n" 
        "You are a medical assistant specialized in the PESI score calculation.\n\n"
        
        "### INSTRUCTIONS: ###\n"
        "Given the following required parameters:\n"
        "Required Parameters: {required_parameters}\n\n"
        
        "And the available patient data:\n"
        "Patient Data Parameters: {patient_data}\n\n"
        
        "1º EVALUATE and identify which 'Patient Data Parameters' has \"missing\" as a value.\n"
        "2º Identify which one of those 'Patient Data Parameters' is related to a parameter in 'Required Parameters'.\n"
        "NOTE that the NAME WILL NOT BE AN EXACT MATCH between the two lists, but PARAMETERS REPRESENT the same characteristic.\n"
        "It is CRUCIAL THAT if the value of a parameter is \"no\", do NOT indicate it as MISSING.\n\n"
        
        "3º Output the list of identified missing parameters that are present in the 'Required Parameters'.\n"
        "4º If there are no missing parameters, respond \"Not Missing Parameters\".\n\n"
        
        "It can be missing parameters in 'Patient Data Parameters' that are not in 'Required Parameters', but you should not include them in the response.\n"
        
        "### OUTPUT FORMAT: ###\n"
        "Use the format:\n"
        "Missing PESI: param1, param2, ... (THE NAME MUST BE THE ONES IN PATIENT DATA)\n"
        "If there is not missing parameter, respond \"Not Missing Parameters \"\n"
        "Reason step by step your actions and conclusions"
    )
    
    llm_input_missing = prompt.format(patient_data=state['patient_data'], required_parameters=required_parameters)

    llm_response_missing = llm.with_structured_output(MissingParametersResponse).invoke(llm_input_missing)

    print(f"###### LLM Missing Response: #### \n\n {llm_response_missing.missing_params}")

    # Parse missing parameters
    missing_parameters = llm_response_missing.missing_params

    # Step 4: Initialize a ParameterRequestMessage for the missing parameters
    # If missing parameters exist, create a ParameterRequestMessage
    if missing_parameters is not None and missing_parameters != "Not Missing Parameters":
        # Filter out parameters that are not in the patient data
        state["messages"].append(ParameterRequestMessage(parameters=missing_parameters))
    else:
        state["messages"].append(AIMessage(content="No missing parameters for PESI and sPESI calculation."))
    return state

##### ---   PESI CALCULATOR NODE   --- #####

def pesi_calculator_node(state: CustomState):
    """
    This node calculates the PESI and sPESI scores using the available patient data and the PESI calculation guidelines.
    It updates the 'metrics' dictionary in the state with the PESI and sPESI scores and classes.
    """
    # Print Current Node
    print("\n#############################################################"
          "\n######## CURRENT NODE: PESI & sPESI CALCULATOR NODE #########"
          "\n#############################################################\n"
        )

    # FIRST: CHECK IF PESI HAS ALREADY BEEN CALCULATED
    if "PESI_Value" in state["metrics"]:
        print("PESI score has already been calculated.")
        state["messages"].append(SystemMessage(content="PESI score has already been calculated."))
        return state

    # Retrieve patient data:
    patient_data = state.get("patient_data")

    # Initialize metrics dictionary if not already present
    if "metrics" not in state:
        state["metrics"] = {}

    ### ---- Calculate PESI ---- ###
    # Retrieve PESI calculation guidelines
    retrieval_query_pesi = "PESI and sPESI calculation formula and scoring"
    docs_content, docs = retrieve(query = retrieval_query_pesi, top_k=1, reranking=False, include_refs=False)

    # Use LLM to calculate the PESI score
    prompt_pesi = (
        "### ROLE: ###\n"
        "You are a medical expert specialized in the PESI score calculation.\n\n"
        
        "### INSTRUCTIONS: ###\n"
        "Using the following patient data: {patient_data}.\n"
        "Reason and calculate the PESI score according to these guidelines:\n"
        "{docs_content}\n\n"
        
        "Reason step by step your actions.\n\n"
        
        "### OUTPUT FORMAT: ###\n"
        "I need you to response strictly on this format:\n"
        "First, explain step by step your actions to calculate the PESI score.\n"
        "At the end, provide the PESI score in this format:\n"
        "\"PESI value: Result PESI Value (Result PESI Class)\". For example, PESI value: 50 (I)\n\n"
        
        "A complete schema could be:\n"
        "(explanation of the actions to calculate the PESI score)\n"
        "-------------------------------------------------------\n"
        "PESI value: 50 (I)\n\n"
        
        "Avoid adding formats like ** or '' when printing the final list.\n\n"
    )
    
    llm_input_pesi = prompt_pesi.format(patient_data=patient_data, docs_content=docs_content)
    
    llm_response_pesi = llm.invoke(llm_input_pesi)
    
    print(f"############### PESI LLM RESPONSE: ################### \n {llm_response_pesi.content}")
    state["messages"].append(llm_response_pesi)

    # Parse the PESI score and class
    pesi_value = []
    pesi_class = []
    for line in llm_response_pesi.content.strip().split("\n"):
        if line.startswith("PESI value:"):
            # Extract the part after "PESI value:"
            content = line.split(":")[1].strip()
            # Extract the PESI value
            pesi_value = content.split("(")[0].strip()
            print(pesi_value)
            # Extract the PESI class
            pesi_class = content.split("(")[-1].strip(")")
            print(pesi_class)
            break

    if pesi_value is not None:
        print(f"\n############### PESI SCORE: ################### \n {pesi_value} ({pesi_class})")
        state["metrics"]["PESI_Value"] = pesi_value
        state["metrics"]["PESI_Class"] = pesi_class
        state["messages"].append(AIMessage(content=f"The patient's PESI score is: {pesi_value} ({pesi_class})."))
        # state["external_messages"].append(AIMessage(content=f"The patient's PESI score is: {pesi_value} ({pesi_class})."))
    else:
        print(f"Unable to calculate PESI score with the provided data.")
        state["messages"].append(AIMessage(content="Unable to calculate PESI score with the provided data."))
        # state["external_messages"].append(AIMessage(content="Unable to calculate PESI score with the provided data."))


    ### ---- Calculate sPESI ---- ###
    # Use LLM to calculate the sPESI score
    prompt_spesi = (
        "### ROLE: ###\n"
        "You are a medical expert specialized in the sPESI score calculation.\n\n"
        
        "### INSTRUCTIONS: ###\n"
        "Using the following patient data: {patient_data}.\n"
        "Reason step by step and calculate the sPESI score according to these guidelines:\n"
        "{docs_content}\n\n"
        
        "### OUTPUT FORMAT: ###\n"
        "I need you to response strictly on this format:\n"
        "First, explain step by step your actions to calculate the sPESI score.\n"
        "At the end, provide the sPESI score in this format:\n"
        "\"sPESI value: Result sPESI Value (Result sPESI class [Low or High])\". For example, sPESI value: 2 (High)\n\n"
        
        "A complete response schema could be:\n"
        "(explanation of the actions to calculate the sPESI score)\n"
        "-------------------------------------------------------\n"
        "sPESI value: 2 (High)\n\n"
        
        "Avoid adding formats like ** or '' when printing the final list.\n\n"
    )
    
    llm_input_spesi = prompt_spesi.format(patient_data=patient_data, docs_content=docs_content)
    
    llm_response_spesi = llm.invoke(llm_input_spesi)

    print(f"\n############### sPESI LLM RESPONSE: ################### \n {llm_response_spesi.content}\n")
    state["messages"].append(llm_response_spesi)

    # Parse the sPESI score
    spesi_score = []
    for line in llm_response_spesi.content.strip().split("\n"):
        if line.startswith("sPESI value:"):
            # Extract the part after "sPESI value:"
            content = line.split(":")[1].strip()
            # Extract the sPESI value
            spesi_score = content.split("(")[0].strip()
            spesi_class = content.split("(")[-1].strip(")")
            print(spesi_score)
            break

    if spesi_score is not None:
        print(f"\n############### sPESI SCORE: ################### \n {spesi_score} ({spesi_class})")
        state["metrics"]["sPESI_Value"] = spesi_score
        state["metrics"]["sPESI_Class"] = spesi_class
        state["messages"].append(AIMessage(content=f"The patient's sPESI score is: {spesi_score} ({spesi_class})."))
        # state["external_messages"].append(AIMessage(content=f"The patient's sPESI score is: {spesi_score}."))
    else:
        print(f"Unable to calculate sPESI score with the provided data.")
        state["messages"].append(AIMessage(content="Unable to calculate sPESI score with the provided data."))
        # state["external_messages"].append(AIMessage(content=f"The patient's sPESI score is: {spesi_score}."))

    return state


#### --- ROEM Parameters Evaluator Node --- ####

def roem_parameters_evaluator_node(state: CustomState):
    """
    This node evaluates the parameters required for the calculation of the Risk of Early Mortality (ROEM) for pulmonary embolism.
    It uses an LLM to identify the necessary parameters based on the available patient data and the PESI metrics.
    If necessary parameters are missing, it will request them from the user invoking the Patient Data Request Tool.
    The node is designed to ensure that all necessary parameters are available before proceeding with the ROEM calculation.
    """
    
    print("\n###############################################################"
          "\n######## CURRENT NODE: ROEM PARAMETERS EVALUATOR NODE #########"
          "\n###############################################################\n"
        )
    
    # FIRST: CHECK IF Risk of Early Mortality HAS ALREADY BEEN CALCULATED
    if "Early_Mortality_Risk" in state["metrics"]:
        print("Risk of Early Mortality has already been calculated.")
        state["messages"].append(SystemMessage(content="Early Mortality Risk has already been calculated."))
        return state

    # Step 1: Retrieve Risk of Early Mortality calculation guidelines
    retrieval_query = "Parameters required to calculate the Severity classification and the Risk of Early Mortality for pulmonary embolism"
    docs_content , docs = retrieve(query = retrieval_query, top_k = 1, reranking = False, include_refs = False)

    # state["messages"].append(SystemMessage(content=f"Context retrieved to extract Risk of Early Mortality classification for pulmonary embolism parameters:\n {docs}"))

    # print(f"###### Contexto Recuperado: #### \n\n {docs_content}")
    
    # Step 2: Use LLM to extract required parameters
    required_params_prompt = (
        "### ROLE: ###\n"
        "You are a medical assistant specialized in the calculation of the Early Mortality Risk for Pulmonary Embolism.\n\n"
        
        "### INSTRUCTIONS: ###\n"
        "This is the available patient data:\n"
        "{patient_data}.\n\n"
        
        "Based on the following guidelines, evaluate how to calculate the Early Mortality Risk for pulmonary embolism and list all the patient data parameters"
        "required to calculate it:\n"
        "{docs_content}\n\n"
        
        "Reason step by step your actions and conclusions.\n\n"
        
        "### OUTPUT FORMAT: ###\n"
        "Your response must be strictly a list with the needed patient data parameters to calculate the given metric."
        "Use this format: param1, param2, param3, ...\n"   
    )
    
    required_params_llm_input = required_params_prompt.format(docs_content=docs_content, patient_data=state['patient_data'])
    
    early_risk_required_params = llm.with_structured_output(RequiredParamsResponse).invoke(required_params_llm_input)

    print(f"###### LLM response: #### \n\n {early_risk_required_params.required_params}")

     # Parse LLM response to extract required parameters
    early_risk_params = early_risk_required_params.required_params
    early_risk_parameters = {"Severity Parameters": early_risk_params}

    print(f"###### Parsed Parameters: #### \n\n {early_risk_parameters}")
    
    # Step 3: Identify missing parameters
    prompt_missing = (
        "### ROLE: ###\n"
        "You are a medical assistant specialized in the calculation of the risk of early mortality for pulmonary embolism.\n\n"
        
        "### INSTRUCTIONS: ###\n"
        "Given the following required parameters:\n" 
        "Required Parameters: {required_params}\n\n"
        
        "And the available patient data:\n"
        "Patient Data Parameters: {patient_data}\n\n"
        
        "1º EVALUATE and identify which 'Patient Data Parameters' has \"missing\" as a value.\n"
        "2º Identify which one of those missing 'Patient Data Parameters' is related to a parameter in 'Required Parameters'.\n"
        "NOTE that the NAME WILL NOT BE AN EXACT MATCH between the two lists, but PARAMETERS REPRESENT the same characteristic.\n"
        "It is CRUCIAL THAT if the value of a parameter is \"no\", do NOT indicate it as MISSING.\n\n"
        
        "3º Output the list of identified missing parameters that are present in the 'Required Parameters'.\n"
        "4º If there are no missing parameters, respond \"Not Missing Parameters\".\n\n"
        
        "It can be missing parameters in 'Patient Data Parameters' that are not in 'Required Parameters', but you should not include them in the response.\n"
        
        "Reason step by step your actions and conclusions\n\n"
        
        "### OUTPUT FORMAT: ###\n"
        "Use STRICTLY this format in the response:\n"
        "Missing: param1, param2, ... (THE NAME MUST BE THE ONES IN PATIENT DATA)\n"
        "If there is not missing parameter, just respond \"Not Missing Parameters\"."
    )

    llm_input_missing_early_risk = prompt_missing.format(required_params=early_risk_parameters, patient_data=state['patient_data'])
    
    llm_response_missing_early_risk = llm.with_structured_output(MissingParametersResponse).invoke(llm_input_missing_early_risk)

    print(f"###### LLM Missing Response: #### \n\n {llm_response_missing_early_risk.missing_params}")

    # Parse missing parameters
    missing_parameters = llm_response_missing_early_risk.missing_params
    
    if missing_parameters is not None and missing_parameters != "Not Missing Parameters":
        state["messages"].append(ParameterRequestMessage(parameters=missing_parameters))
    else:
        state["messages"].append(AIMessage(content="No missing parameters for the calculation of the Early Mortality Risk."))
    return state


##### ---  Calculate ROEM Score   --- #####

def roem_calculator_node(state: CustomState):
    """
    This node calculates the Risk of Early Mortality (ROEM) for pulmonary embolism using the available patient data and the PESI metrics.
    It updates the 'metrics' dictionary in the state with the Risk of Early Mortality level.
    """
    
    print("\n#####################################################"
          "\n######## CURRENT NODE: ROEM CALCULATOR NODE #########"
          "\n#####################################################\n"
        )

    # FIRST: CHECK IF Risk of Early Mortality HAS ALREADY BEEN CALCULATED
    if "Early_Mortality_Risk" in state["metrics"]:
        print("Risk of Early Mortality has already been calculated.")
        state["messages"].append(SystemMessage(content="Risk of Early Mortality has already been calculated."))
        return state

    # Retrieve patient data and metrics
    patient_data = state.get("patient_data")
    metrics = state.get("metrics")

    # Ensure PESI metrics are available
    pesi_value = metrics.get("PESI_Value")
    pesi_class = metrics.get("PESI_Class")
    spesi_value = metrics.get("sPESI_Value")

    if pesi_value is None or pesi_class is None:
        state["messages"].append(AIMessage(content="Risk of Early Mortality cannot be calculated because PESI metrics are missing."))
        state["external_messages"].append(AIMessage(content="Risk of Early Mortality cannot be calculated because PESI metrics are missing."))
        return state

    # Retrieve Risk of Early Mortality calculation guidelines
    retrieval_query_early_risk = "Severity classification and Risk of Early Mortality for pulmonary embolism"
    docs_content , docs = retrieve(query = retrieval_query_early_risk, top_k=1, reranking=False, include_refs=False)  

    # Use LLM to classify the Risk of Early Mortality of PE
    prompt_early_risk = (
        "### ROLE: ###\n"
        "You are a medical assistant specialized in the calculation of the risk of early mortality for pulmonary embolism.\n\n"
        
        "### INSTRUCTIONS: ###\n"
        "Using the following patient data:\n"
        "{patient_data}\n\n"
        
        "And PESI and sPESI metrics:\n "
        "PESI Value = {pesi_value}, PESI Class = {pesi_class}, sPESI Value = {spesi_value}\n\n"
        
        "Reason step by step and classify the risk of early mortality of PE according to these guidelines:\n"
        "{docs_content}\n\n"
        
        "Take into account that your response will be reviewed by a medical expert. Be clear and precise.\n"
        "It is crucial to ensure patient safety, so be careful when assuming that some variables can be assumed as negative.\n"
        "If you are not sure about how to classify a variable, please indicate it in your response. You could indicate an inteval of values if necesarry.\n"
        
        "Reason step by step your actions and conclusions.\n\n"
        
        "### CONSIDERATIONS: ###\n"
        "It is CRUCIAL that if you need to use numerical ratios, you have to interpret them as indicative of a POSITVE CONDITION when their VALUE is = or > than 1.\n\n"
        
        "### OUTPUT FORMAT: ###\n"
        "I need you to response strictly on this format:\n"
        "First, explain step by step your actions to calculate the risk of early mortality.\n"
        "At the end, provide the obtained Risk of Early Mortality level on this format:\n"
        "Risk of Early Mortality level: Obtained level. For example, 'Risk of Early Mortality level: Low'.\n\n"
        
        "A complete response schema could be:\n"
        "(explanation of the actions to classify the Risk of Early Mortality level)\n"
        "-------------------------------------------------------\n"
        "Risk of Early Mortality level: Low\n\n"
        
        "AVOID adding formats like ** or '' when printing the final list."
    )

    llm_input_early_risk = prompt_early_risk.format(patient_data=patient_data, pesi_value=pesi_value, pesi_class=pesi_class, spesi_value=spesi_value, docs_content=docs_content)

    llm_response_early_risk = llm.invoke(llm_input_early_risk) 
    
    print(f"############### Risk of Early Mortality LLM RESPONSE: ################### \n {llm_response_early_risk.content}")
    state["messages"].append(llm_response_early_risk)

    # Parse the Risk of Early Mortality level
    early_risk_level = None
    for line in llm_response_early_risk.content.strip().split("\n"):
        if line.startswith("Risk of Early Mortality level"):
            early_risk_level = line.split(":")[1].strip()
            break

    if early_risk_level:
        print(f"\n############### SEVERITY LEVEL: ################### \n {early_risk_level}")
        state["metrics"]["Early_Mortality_Risk"] = early_risk_level
        state["messages"].append(AIMessage(content=f"The patient's Risk of Early Mortality of PE is classified as: {early_risk_level}."))
        # state["external_messages"].append(AIMessage(content=f"The patient's Risk of Early Mortality of PE is classified as: {early_risk_level}."))
    else:
        print(f"Unable to classify Risk of Early Mortality of PE with the provided data.")
        state["messages"].append(AIMessage(content="Unable to classify the Risk of Early Mortality of PE with the provided data."))
        # state["external_messages"].append(AIMessage(content="Unable to classify the Risk of Early Mortality of PE with the provided data."))

    return state