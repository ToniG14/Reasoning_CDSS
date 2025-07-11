#################################
##### ---    IMPORTS    --- #####
#################################

from langchain_core.messages import AIMessage
from src.services.hallucination_detector import HallucinationDetectorFactory
from src.custom_config.state_schema import CustomState
from src.custom_config.custom_messages import ClinicalCaseEvaluationReport, UserQuery, ParameterRequestMessage, ToolRequest
from src.services.tools import retrieval_tool
from src.llm_config import *

#############################################
##### ---    CLASSES & FUNCTIONS    --- #####
#############################################

#### ---   Retrieval Tool   --- ####

def retrieval_tool_node_2(state: CustomState):
    """
    This node use the invoked Retrieval Tool to retrieve information from the PE guidelines based on a query.
    This node is invoked when the LLM agent decides to call the Retrieval Tool to gather information from the PE guidelines.
    It extracts the query and parameters from the tool call message, invokes the Retrieval Tool, and appends the retrieved information to the state.
    The retrieved information is then used in the next node to generate the clinical case evaluation report.
    """
    
    print("\n######################################################")
    print("######## CURRENT NODE: RETRIEVAL TOOL NODE ###########")
    print("######################################################\n")
    
    ## GET RETRIEVAL PARAMETERS
    # Get the tool call message
    tool_call = None
    for message in reversed(state["messages"]):
        if isinstance(message, ToolRequest):
            tool_call = message
            break
    
    # Extract query
    query = tool_call.args.get("query")
    
    # Extract top_k
    top_k = int(tool_call.args.get("top_k"))
    
    # Extract top_k_rerank
    top_k_rerank = int(tool_call.args.get("top_k_rerank"))

    # Create the retrieval arguments dictionary
    retrieval_args = {
        "query": query,
        "top_k": top_k,
        "top_k_rerank": top_k_rerank,
        "reranking": True,
        "include_refs": True
    }

    # Invoke the tool
    docs = retrieval_tool.invoke(retrieval_args)
    
    # Print results
    print(f"Tool call response: ")
    metadata = [f"{doc.metadata}\n" for doc in docs]
    print(f"Retrieved Docs in tool: {metadata}")
    print(f"Retrieved information: {docs}")
    
    # Create message to append the state
    message = f"The information for the query '{query}' has been retrieved."
    print(f"\n##### Tool Call Response #####\n{message}\n")
    
    # Save the response in the state
    state["messages"].append(AIMessage(content=message))
    state["retrieval_queries"].append(query)
    state["retrieved_information"].append(docs)
    
    return state

#### ---   Clinical Case Evaluator Node   --- ####

def clinical_case_evaluator_node(state: CustomState):
    """
    This node evaluates the clinical case of a patient with suspected or confirmed Pulmonary Embolism (PE).
    It analyzes patient data, retrieves relevant information from the PE guidelines, and prepares for generating the clinical case evaluation report.
    
    The node uses an LLM to reason step-by-step, considering patient metrics, clinical data, and previous chat history.
    It can call the Retrieval Tool to fetch specific guidelines and the Missing Patient Data tool to request additional clinical data.
    The node operates in a multi-turn loop, allowing for iterative reasoning and information retrieval until the final report can be prepared.
    
    The clinical case evaluation report will be generated in the next node.
    """
    
    print("\n##############################################################"
          "\n######## CURRENT NODE: CLINICAL CASE EVALUATOR NODE ##########"
          "\n##############################################################\n")

    ## PATIENT DATA AND METRICS
    metrics = state.get("metrics", {})
    patient_data = state.get("patient_data", {})
    
    ## USER's QUERY
    user_query = None
    for message in reversed(state["external_messages"]):
        if isinstance(message, UserQuery):
            user_query = message.content
            break

    if not user_query:
        raise ValueError("No UserQuery found in external messages.")

    # print(f"##### User Query: #####\n{user_query}\n")

    ## CHAT HISTORY
    chat_history = state["external_messages"]
    
    ## PE in CTPA
    PE_ctpa = state.get("patient_data").get("CTPA").get("Pulmonary embolism")
    
    ## RETRIEVED INFORMATION
    
    # Get the retrieved information from the state
    if state["retrieved_information"]:
        formatted_info = []
        for query, docs in zip(state["retrieval_queries"], state["retrieved_information"]):
            section_lines = [f"For the query: ({query})\n",
                            "The retrieved information is:"]
            for i, doc in enumerate(docs, start=1):
                section_lines.append(f"- Doc {i}: ({doc.page_content})")
            formatted_info.append("\n\n".join(section_lines))
        retrieved_information = "\n\n".join(formatted_info)
    else:
        retrieved_information = "No information retrieved yet."
    
    retrieval_queries = state["retrieval_queries"]
    
    print(f"##### Processed Retrieval Queries #####\n{retrieval_queries}\n")
    
    ## Remaining Retrieval Calls
    remain_retrieval_calls = 4 - len(state["retrieval_queries"])
    print(f"##### Remaining Retrieval Calls #####\n{remain_retrieval_calls}\n")
    
    ## Use the LLM to generate a step
    prompt = (
        "# ROLE:\n"
        "You are a medical assistant supporting physicians in managing Pulmonary Embolism (PE) patients.\n"
        "Your role is to reason step-by-step to analyze patient data to determine the state of the patient, retrieve information from the PE guidelines and then provide the best recommendations for the current selected patient.\n"
        "⚠️ Final recommendations will be generated in the next node.\n\n"
        
        "---\n\n"
        
        "# AVAILABLE TOOLS:\n"
        "To accomplish you final objetive, you have access to the following tools:\n\n"
        
        "## 1. 'Retrieval Tool':\n"
        "This tool allows you to retrieve relevant paragraphs from the Pulmonary Embolism Guidelines based on a query to a vectorstore.\n"
        "The Pulmonary Embolism Guidelines is a document with recommendations on how to diagnose and treat patients with PE. The guidelines has been splitted in paragraphs, so one passage corresponde to a paragraph.\n\n"
        
        "The arguments of the tool are:\n\n"
        
        "- query (str): The query to search for passages (paragraphs) in the vectorstore.\n"
        "- top_k (int): The number of paragraphs to retrieve.\n"
        "- top_k_rerank (int): Number of paragraphs to keep after reranking.\n"

        "Returns:\n\n"
        
        "- final_results (list): List of all retrieved documents.\n\n"
        
        "### To call this tool you can use the following format:\n\n"
        
        "Retrieval Call: query = (query to retrieve info), top_k = (number of paragraphs to retrieve), top_k_rerank = (number of paragraphs to keep after reranking)\n"
        "Adequate the queries to the vectorstore to retrieved the different information needed. If you use the same query, you will receive the same information.\n\n"
        
        "THERE IS A LIMIT IN THE NUMBER OF RETRIEVAL CALLS THAT YOU CAN DO. THIS LIMIT IS 4 RETRIEVAL CALLS. THE NUMBER OF REMAINING CALLS TO THE RETRIEVAL TOOL IS {remain_retrieval_calls}. When this number is 0, you must not use the retrieval tool anymore.\n\n"
        "⚠️ Important: You are strictly forbidden to exceed the limit of calls to the information retrieval tool."
        
        "## 2. 'Missing Patient Data':\n"
        "This tool allows you to request missing patient clinical data that is needed to make a safe and accurate recommendation.\n\n"
        "You can only request for parameters which value is 'missing' in the 'PATIENT DATA' section.\n"
        "The value 'missing' is used to represent that this parameters has not been registered but the information is available, so it can be requested with a call to this tool.\n"
        "The value 'Not Available' is used to represent that those clinical parameters correspond to laboratory and imaging test and can be performed now so that info is not available now. DO NOT REQUEST FOR PARAMETERS WHICH VALUE IS 'NOT AVAILABLE'.\n"
        
        "### To call this tool, you can use the following format:\n"
        "Missing Patient Data: param1, param2, ...\n"
        "The parameters must be the ones in the 'PATIENT DATA' section. Those are the only available data. Do not request for parameters that are not present in 'PATIENT DATA'.\n\n"
        
        "---\n\n"
        
        "# CLINICAL OBJECTIVE:\n"
        "There are three possible clinical scenarios:\n"
        "1. Confirmed PE: In this case, you must provide treatment recommendations.\n"
        "2. Suspected PE + Stable Patient: In this case, you must provide diagnostic recommendations.\n"
        "3. Suspected PE + Unstable Patient: In this case, you must provide both diagnostic and treatment recommendations.\n\n"
        
        "⚠️ Important: The patient parameter 'Pulmonary embolism' in the 'CTPA' category indicates whether PE has been confirmed via CTPA imaging. If this field is missing, **do not assume the patient is PE-negative**.\n"  
        "Pulmonary embolism in CTPA: {PE_CTPA}\n\n"
        
        "# INSTRUCTIONS: (IMPORTANT)\n\n"
        "You operate in a **multi-turn loop**, reasoning across multiple steps.\n"
        "Each response is part of an ongoing process. Do not attempt to complete everything in one message.\n"
        "Use the **CHAT HISTORY** to continue from where you left off.\n\n"
        
        "You may:\n"
        "- Analyze the patient state and data.\n"
        "- Identify gaps in knowledge.\n"
        "- Call the Retrieval Tool (to query PE guidelines).\n"
        "- Call the Missing Patient Data tool (to request necessary missing data).\n"
        "- Repeat these steps across several turns until you are ready to trigger the final recommendation step.\n"
        "- Proceed with patient clinical data which value is 'Not Available', understanding that it can not be requested with the 'Missing Patient Data' tool. In the case of this parameters are really relevant for the clinical case, you can recommend the physician to perform the imaging test or laboratory test necessary to gather the parameters for him (Just a recommendation, do not request the parameter)\n\n."

        "⚠️ Your response in each turn must be one of the following:\n"
        "   - Just reasoning: Useful to plan you next steps, understand patient's state and clinical case, analyze the patient data and information retrieved and you previous thoughts,...\n"
        "   - Reasoning and calling the 'Retrieval Tool': Useful to analyze what information you need to retrieve from the PE guidelines, and how to desing proper queries to retrieve the information you need.\n"
        "   - Reasoning and calling the 'Missing Patient Data': Useful to analyze what information about patient clinical data you need to request to the physician.\n\n"
        
        "You must retrieve information from the PE guidelines about how to diagnose and treat the specific patient you are dealing with, clinical tests to diagnose PE, how to interpret the results of those tests, contraindications in the recommended treatments and what to do in those cases, and about specific treatments like drugs and drug doses(important if providing treatment recommendations).\n\n"
        
        "The instructions for the tools are in the 'AVAILABLE TOOLS' section.\n\n"

        "## POSSIBLE ACTIONS:\n"
        "- You can proceed with more than one reasoning step before generating the final recommendations.\n"
        "- You can call the 'Retrieval Tool' and the 'Missing Patient Data' tool multiple times but not exceed the limits.\n"
        "- Regarding the 'Retrieval Tool':\n"
        "   - Avoid using the similar queries, because that will retrieve the same information from the guidelines.\n"
        "- Regarding the 'Missing Patient Data' tool:\n"
        "   - You will have to work with 'Not Available' values in patient's clinical data and YOU CAN'T REQUEST FOR 'NOT AVAILABLE' CLINICAL DATA. You must account for that in your reasoning.\n"
        "   - You only can request for parameters which value in the 'PATIENT DATA' section is 'missing'. Do not request for parameters which value is 'Not Available'. In these case, consider recommend the physician to perform the imaging test or laboratory test to obtain the information if the parameters are really relevant for the clinical case.\n"
        "   - You only can request a specific parameter once. If you request a parameter, you can not request it again.\n"
        "   - Avoid requesting parameters that are not crucial for generating the recommendations.\n"
        "- You can also reason multiple times before generating the final recommendations.\n\n"

        "- Only when you have gathered all the required information (from PE guidelines and missing patient data) to provide the best recommendations for the patient, you are ready to generate the final recommendations.\n" 
        "- To indicate that you are finally prepared to generate the final recommendations, you should end your final reasoning message with a final line containing exactly: 'PREPARE RECOMMENDATIONS'\n"
        "- That will set the flag 'SHOULD_OUTPUT_FINAL_RECOMMENDATION' to 'True'.\n"
        "- Do NOT include 'PREPARE FINAL MESSAGE' at the end of every reasoning step. Only declare it once, at the end of your LAST reasoning turn.\n\n"
        
        "- Do not generate the final recommendation until you are confident you have all necessary information from PE guidelines, patient clinical data and previous reasoning.\n"
        
        "⚠️ Important: You are strictly forbidden to exceed the limit of calls to the information retrieval tool. You must not use the retrieval tool anymore when the number of remaining calls is 0.\n\n"
     
        "# PATIENT DATA:\n"
        "In patient data, you will find different categories of clinical information about the patient.\n"
        "Inside each category, you will find the parameters that can be registered for that category but not all of them are always registered or available.\n"
        "The parameters that are not registered yet are marked as 'missing'. You can request for those parameters using the 'Missing Patient Data' tool.\n"
        "The parameters that are not available are marked as 'Not Available'. This not available parameters correspond to imaging tests and laboratory tests that have not been performed yet so you can NOT request for those parameters because they are not available.\n"
        
        "Patient data:\n {patient_data}\n\n"
        
        "In available clinical data, you will find 3 clinical metrics that assess the risk and the severity of the PE of the patient, take into account the values of these metrics to make the recommendations:\n"
        "Available Clinical Metrics:\n {metrics}\n\n"
        
        "# CONTRAINDICATIONS:\n"
        "Always verify the patient's data against absolute and relative contraindications for thrombolysis.\n"
        "If any contraindications are present, you must search for what to do in case of contraindications in the PE guidelines. Avoid conflicting treatments and retrieve alternatives from the guidelines.\n"
        "The conditions that must requiere attention for contraindications are the patient's clinical data in the 'Absolute Contraindications for Thrombolysis' and 'Relative Contraindications for Thrombolysis' in the 'PATIENT DATA' section.\n\n"
        
        "---\n\n"
        
        "# CHAT HISTORY:\n"
        "Chat history with the previous reasoning and step made:\n"
        "{chat_history}\n\n"
        "It is really important that you consider the indications that you have prepare to yourself in the previous step.\n\n"
        
        "# RETRIEVED INFORMATION:\n"
        "PE Guideline paragraphs retrieved so far:\n"
        "{docs_content}\n\n"
            
        "---\n\n"
        
        "# OUTPUT FORMAT: (CRUCIAL)\n"
        "Your response must be strictly in the following format:\n\n"
        
        "1. Reason clearly and explain your current thought process.\n"
        "2. If needed, call a tool. Insert a line with exactly:\n"
        "   -----------------------\n"
        "   Followed by the tool call in one of these formats:\n"
        "   - Retrieval Call: query = (query), top_k = (number), top_k_rerank = (number)\n"
        "   - Missing Patient Data: param1, param2, ...\n\n"
        "3. If and only if you are ready for the final recommendation step, end your message with:\n"
        "   PREPARE RECOMMENDATIONS\n"
        
        "When you have finish all the reasoning turns and you have stablish that you are ready to generate the recommendations, you must write at the end of your final message on a new line: 'PREPARE RECOMMENDATIONS'\n"
        "Do not include 'PREPARE RECOMMENDATIONS' in every reasoning step. Only declare it once, at the end of your LAST reasoning turn.\n\n"
        "This will trigger the generation of the final recommendation, so just do it when you have finish all the reasoning steps.\n\n"
        
        "# REMEMBER:\n"
        "Your reasoning will be reviewed by a medical expert. Be clear and precise.\n"
        "Under no circumstances you can request for parameters that are 'Not Available'. When a patient clinical parameter is 'Not Available', it means that the test to obtain it has not been performed yet, so you can not request it, you need to recommend in the final recommendation to perform the required test to obtain it.\n"
        "If you are in the case that these patient clinical parameters (Those with 'Not Available' values) are important, your recommendations should go in the direction of performing the imaging test or laboratory test to obtain the information.\n"
        "Remember that you are not only recommending treatments but also diagnostic tests if needed.\n"
)

    llm_input = prompt.format(
        patient_data=patient_data,
        metrics=metrics,
        PE_CTPA=PE_ctpa,
        chat_history=chat_history,
        docs_content=retrieved_information,  
        remain_retrieval_calls=remain_retrieval_calls,
    )

    # print(f"##### LLM Input #####\n{llm_input}\n")

    llm_response_recommendation = llm.invoke(llm_input)
    # llm_response_recommendation = llm_reasoning.invoke(llm_input)

    print(f"##### LLM Response #####\n{llm_response_recommendation.content}\n")
    state["messages"].append(llm_response_recommendation)


    # 3. Parse the LLM response
    content = llm_response_recommendation.content.strip()

    if "Retrieval Call:" in content:
        print("##################\n Retrieval Call \n##################\n")
        # Extrae la primera línea que contiene la llamada a la herramienta
        retrieval_line = content.split("Retrieval Call:", 1)[1].split("\n", 1)[0].strip()
        args = {}
        for arg in retrieval_line.split(","):
            if "=" in arg:
                arg_name, arg_value = arg.split("=", 1)
                args[arg_name.strip()] = arg_value.strip().strip('"')
        state["messages"].append(
            ToolRequest(content=content, args=args)
        )
        print(f"###### Retrieval Call: ###### \n{retrieval_line}\n")
    
    elif "Missing Patient Data:" in content:
        print("###########################################################\n Requiered Missing Parameter to Generate the Recommendations \n###########################################################\n")
        missing_line = content.split("Missing Patient Data:", 1)[1].split("\n", 1)[0].strip()
        missing_data = missing_line
        state["messages"].append(
            ParameterRequestMessage(parameters=missing_data.split(", "))
        )
        print(f"Missing Patient Data: \n{missing_data}\n")

    else:
        print("###############\n Reasoning Step \n###############\n")
        state["messages"].append(AIMessage(content=content))
        
    # Check if the final message preparation is indicated
    # This is used to trigger the final recommendation generation in the next step
    if 'PREPARE RECOMMENDATION' in content:
        print("- DETECTED PREPARE RECOMMENDATION\n")
        state['prepare_final_message'] = True

    return state



##### ---   Display Recommendation Node   --- #####

def clinical_case_report_generator_node(state: CustomState):
    """
    This node generates the final clinical case report based on the previously gathered information, patient data, and user query.
    It analyzes the chat history, retrieved information, and patient metrics to produce a well-structured, personalized clinical case report for the physician.
    The node uses an LLM agent to synthesize the information and generate the final report.
    It operates in a single turn, producing the final output based on the information available in the state.
    
    The report will include:
        - Patient State
        - Diagnosis
        - Recommendations (Diagnostic and/or Treatment)
        - Summary of the report
    """
    
    print("\n#####################################################################"
          "\n######## CURRENT NODE: CLINICAL CASE REPORT GENERATOR NODE ##########"
          "\n#####################################################################\n"
          )

    ## PATIENT DATA AND METRICS
    metrics = state.get("metrics", {})
    patient_data = state.get("patient_data", {})
    
    ## USER's QUERY
    user_query = None
    for message in reversed(state["external_messages"]):
        if isinstance(message, UserQuery):
            user_query = message.content
            break

    if not user_query:
        raise ValueError("No UserQuery found in external messages.")

    # print(f"##### User Query: #####\n{user_query}\n")

    ## CHAT HISTORY
    chat_history = state["external_messages"]
    
    ## PE in CTPA
    PE_ctpa = state.get("patient_data").get("CTPA").get("Pulmonary embolism")
    
    ## RETRIEVED INFORMATION
    
    # Get the retrieved information from the state
    if state["retrieved_information"]:
        formatted_info = []
        for query, docs in zip(state["retrieval_queries"], state["retrieved_information"]):
            section_lines = [f"For the query: ({query})\n",
                            "The retrieved information is:"]
            for i, doc in enumerate(docs, start=1):
                section_lines.append(f"- Doc {i}: ({doc.page_content})")
            formatted_info.append("\n\n".join(section_lines))
        retrieved_information = "\n\n".join(formatted_info)
    else:
        retrieved_information = "No information retrieved yet."
        
    retrieval_queries = state["retrieval_queries"]
    
    print(f"##### Processed Retrieval Queries #####\n{retrieval_queries}\n")
    
    ## Use the LLM to generate a step
    prompt = (
        "# ROLE:\n"
        "You are a medical assistant responsible for delivering the final clinical recommendation for a patient with suspected or confirmed Pulmonary Embolism (PE).\n"
        "Your job is to analyze all previously gathered information and produce a well-structured, personalized best recommendation for the current patient that will be shown directly to the physician.\n\n"
        
        "Note: Your response will be delivered to a physician.\n"
        "Therefore, it is essential to maintain a clinical tone, use precise medical language, and apply the appropriate level of caution and responsibility required in clinical decision-making.\n\n"
        
        "# FINAL OBJECTIVE:\n"
        "Get the all the information in 'CHAT HISTORY' (step-by-step clinical reasoning already performed), the 'RETRIEVED INFORMATION' (paragraphs from PE guidelines) and 'PATIENT DATA'(clinical parameters and contraindications) to provide the best recommendations for the current selected patient.\n"
        "This final recommendations must follow the 'CASE LOGIC' indications.\n\n"
        
        "# PATIENT DATA:\n"
        "You will work with unknown clinical parameters. Consider this wh recommendations.\n"
        "Patient data:\n {patient_data}\n\n"
        
        "Available Clinical Metrics:\n {metrics}\n\n"
        
        "# CONTRAINDICATIONS:\n"
        "Review the sections 'Absolute Contraindications for Thrombolysis' and 'Relative Contraindications for Thrombolysis' in the patient data (PATIENT DATA section).\n"
        "You must avoid recommending any actions that conflict with those contraindications.\n"
        "If such conflicts exist, propose safe alternatives from the guideline information.\n\n"
        
        "---\n\n"
        
        "# CHAT HISTORY:\n"
        "Chat history with the previous reasoning and step made:\n"
        "{chat_history}\n\n"
        "It is really important that you consider the indications that you have prepare to yourself in the previous step.\n\n"
        
        "# RETRIEVED INFORMATION:\n"
        "Retrieved Paragraph from the PE guidelines:\n"
        "{docs_content}\n\n"

        "---\n\n"
        
        "# INSTRUCTIONS:\n"
        "You are now delivering the final output. Your job is to produce a high-quality clinical recommendation.\n\n"

        "⚠️ IMPORTANT: Base all content strictly on:\n"
        "- The patient's actual data.\n"
        "- Your prior reasoning in the chat history.\n"
        "- The paragraphs retrieved from the PE guidelines.\n"
        "Do not introduce new assumptions or external knowledge.\n\n"

        "## CASE LOGIC (CRUCIAL):\n"
        "You will find 3 different clinical situations:\n"
        "1. Confirmed PE: In this case, you must provide treatment recommendations.\n"
        "2. Suspected PE + Stable Patient: In this case, you must provide diagnostic recommendations.\n"
        "3. Suspected PE + Unstable Patient: In this case, you must provide both diagnostic and treatment recommendations.\n\n"
        
        "## QUALITY REQUIREMENTS:\n"   
        "⚠️ Your recommendations must:\n\n"
        "- The recommendations must be specific for the patient, so you must use the patient data to generate the recommendations.\n"
        "- The recommendations must be really specific to the current clinical case, you must be really specific on details on the recommendations of diagnosis test or treatment. For example, when recommending drug treatment, you must indicate a specific drug and the specific drug dosage for that patient.\n"
        "- Be medically accurate, evidence-based, and completely safe for the patient.\n"
        "- Clearly explain the reasoning behind each recommendation based on the guidelines.\n" 
        "- Highlight any necessary precautions or risks explicitly for each recommendation. If you find any contraindication, generate alternatives for the patient to avoid that contraindications.\n"
        "- Be precise: include exact drug names and dosages, diagnostic test names, etc.\n"
        "- These recommendations are intended for physicians, not patients. Always provide a reasoned, clear, and cautious clinical rationale behind each recommendation.\n\n"

        " ⚠️ Some clinical parameters of the patient can be 'Not available'. Those are related with imaging and lab test, so you must account for that and consider to recommend to perform those tests if the parameters are relevant for the diagnosis or treatment of the current patient (Just if the parameters are really relevant).\n\n"

        "## HIGH-RISK DECISION CAUTION: \n\n"

        "When making final recommendations, you must be extremely cautious when suggesting that:\n\n"

        "- A diagnostic test (e.g., imaging, blood test) is **not necessary**.\n"
        "- A therapeutic intervention (e.g., anticoagulation, hospitalization) is **not required**.\n"
        "- The patient is **unlikely to have PE** or **does not require action** due to seemingly low severity.\n\n"

        "⚠️ You must **not** exclude diagnostic or therapeutic options based only on inferred low probability or apparent mild symptoms — **unless this is explicitly supported by the clinical guidelines retrieved** and aligned with the patient's full data profile.\n\n"

        "If the guideline content does not clearly support the omission of a diagnostic test or treatment in a similar patient context, your recommendation must reflect this uncertainty and include a follow-up or safety net approach (e.g., monitoring, reassessment, shared decision-making).\n"
        "Always assume that **under-treatment may have fatal consequences**, and avoid introducing clinical risk by over-relying on subjective or indirect inferences.\n"

        "Prioritize safety, evidence, and transparency at all times.\n\n"

        "# OUTPUT FORMAT:\n"
        "Respond strictly using the following structure:\n\n"
        
        "FINAL RECOMMENDATIONS:\n"
        "- Patient State: (Concise description of patient profile and relevant clinical data)\n\n"
        
        "- Diagnosis: (Clearly state the diagnosis status. If confirmed, say so. If suspected, explain why, and whether the patient is stable or unstable.)\n\n"
        
        "- Recommendations: (Display the recommendations for the patient as described in the 'INSTRUCTIONS' section.)(Try to indicate the level and class of each recommendation if available)\n"

        "- Summary: (Summarize key points of your recommendation and any urgent notes for the physician.)"
        
        "# REMEMBER:\n"
        "Your recommendations could be about diagnostic tests, treatments, or both. In some cases will be necessary to recommend imaging or laboratory tests, given those values may be missing and could be relevant for the clinical case.\n"
)

    llm_input = prompt.format(
        patient_data=patient_data,
        metrics=metrics,
        PE_CTPA=PE_ctpa,
        chat_history=chat_history,
        docs_content=retrieved_information,
    )

    # print(f"##### LLM Input #####\n{llm_input}\n")

    llm_response_recommendation = llm.invoke(llm_input)

    print(f"##### LLM Response #####\n{llm_response_recommendation.content}\n")
    state["messages"].append(llm_response_recommendation)


    # 3. Parse the LLM response
    recommendation_content = llm_response_recommendation.content.strip()

    # Extract hallucinations
    hallucination_detector = HallucinationDetectorFactory.load_detector("RefChecker")
    detected_hallucinations = hallucination_detector.detect_hallucinations(llm_response_recommendation.content, docs)

    print(f"\n###### Detected Hallucinations: ###### \n")
    print(detected_hallucinations)

    recommendation_message = ClinicalCaseEvaluationReport(
        content=recommendation_content,
        sources=docs,
        query=user_query,
        hallucinations=detected_hallucinations
    )
    state["messages"].append(recommendation_message)
    state["external_messages"].append(recommendation_message)
    state["clinical_case_evaluation_reports"].append(recommendation_message)

    return state
