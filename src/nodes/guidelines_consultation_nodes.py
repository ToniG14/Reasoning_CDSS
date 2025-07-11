#################################
##### ---    IMPORTS    --- #####
#################################
from langchain_core.messages import ToolMessage

from src.custom_config.custom_messages import *
from src.services.tools import retrieval_tool
from src.custom_config.state_schema import CustomState
from src.services.hallucination_detector import HallucinationDetectorFactory
from src.custom_config.custom_messages import UserQuery, GuidelinesConsultationResponse
from src.llm_config import *

#############################################
##### ---    CLASSES & FUNCTIONS    --- #####
#############################################

#### --- Tool Node --- #####

retr_tools = [retrieval_tool]
tools_by_name = {tool.name: tool for tool in retr_tools}
    
def retrieval_tool_node(state: CustomState):
    """
    This node uses the invoked Retrieval Tool to retrieve information from the PE guidelines based on a query.
    This node is invoked when the LLM agent decides to call the Retrieval Tool to gather information from the PE guidelines.
    It extracts the query and parameters from the tool call message, invokes the Retrieval Tool, and appends the retrieved information to the state.
    The retrieved information is then used in the next node to generate the guidelines consultation response.
    """
    
    print("\n######################################################")
    print("######## CURRENT NODE: RETRIEVAL TOOL NODE #############")
    print("########################################################\n")
    
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        # Invoke the tool
        docs = tool.invoke(tool_call["args"])
        # Print results
        print(f"Tool call response: ")
        metadata = [f"{doc.metadata}\n" for doc in docs]
        print(f"Retrieved Docs in tool: {metadata}")
        print(f"Retrieved information: {docs}")
        
        # Extract Retrieval Query
        query_used = tool_call["args"].get("query")
        
        # Save the response in the state
        state["messages"].append(ToolMessage(content=docs, tool_call_id=tool_call["id"]))
        state["retrieved_information"].append(docs)
        state["retrieval_queries"].append(query_used)
    return state


##### ---   Query Solver Node   --- #####

def query_solver_node(state: CustomState):
    """
    This node analyzes the physician's questions and provides detailed and reasoned answers based on the information retrieved from the guidelines.
    It uses the Retrieval Tool to gather relevant information and then synthesizes it into a comprehensive response.
    This node is operated by an LLM agent that reasons step-by-step, considering the user query, retrieved information, and chat history.
    The node is designed to handle multi-turn interactions, allowing the LLM to call the Retrieval Tool multiple times if necessary.
    It finally generates a detailed and reasoned answer to the user query based on the retrieved information.
    """

    print("\n####################################################"
          "\n######## CURRENT NODE: QUERY SOLVER NODE ###########"
          "\n####################################################\n"
          )

    # Retrieve the last user query
    user_query = None
    for message in reversed(state["messages"]):
        if isinstance(message, UserQuery):
            user_query = message.content
            break
    
    # Format into prompt
    prompt = (
        " # ROLE:\n"
        "You are a medical assistant specializing in Pulmonary Embolism (PE). Your role is to analyze the physician's questions and provide detailed and reasoned answers based on the information retrieved from the guidelines.\n"
        "You have access to a retrieval tool that allows you to search for information in the guidelines. Your task is to use this tool to gather relevant information and then synthesize it into a comprehensive response.\n\n"
        
        "Note: Your response will be delivered to a physician.\n"
        "Therefore, it is essential to maintain a clinical tone, use precise medical language, and apply the appropriate level of caution and responsibility required in clinical decision-making.\n\n"

        
        "# INSTRUCTIONS:\n"
        "Use the tool to retrieve all the information you need from the PE guidelines to answer the question. Adapt the tool arguments to your needs to retrieve relevant information for the query.\n"
        "If you use the same query in the tool call, it will be returned the same information from the guidelines. DO NOT USE THE SAME QUERY IN MORE THAN ONE TOOLCALL.\n"
        "When you have all the information you need, answer the question in a detailed and reasoned manner.\n\n"
        
        "- Your inner execution plan must be:\n"
        "1º Use the tool to retrieve information from the guidelines.\n"
        "2º Use the retrieved information to answer the question.\n\n"
                
        "Your answers must strictly adhere to the context. DO NOT INVENT or add information not explicitly mentioned in the context.\n"
        "Under no circumstances assume anything, do not assume that because information about some parameter or characteristic is missing that it is not being produced or does not exist.\n" 
        "If you need to retrieve more info, do it and be specific about what you need. If you use the same query in the tool call, it will be returned the same information from the guidelines. Do not use two times the same query.\n\n"

        "# User Query:\n"
        "User Query: {user_query} \n"

        "# RETRIEVED INFORMATION: \n"
        "The current information retrieved from the guidelines is:\n"
        "{retrieved_info}\n\n"
        
        "# ACTIONS HISTORY:\n"
        "Here you can check how many tool calls you have done and the information retrieved from the guidelines:\n"
        "Here you have the remaining number of calls to the retrieval tool: '{remaining_tool_calls}'. WHEN YOU REACH 0, YOU CAN NOT USE THE RETRIEVAL TOOL ANYMORE, YOU MUST ANSWER.\n"
        "Here you have the previous employed retrieval queries: {retrieval_queries}. DO NOT REPEAT THE SAME QUERY IN MORE THAN ONE TOOLCALL.\n\n"
        "You do not need to reach the limit of tool calls to answer the question.\n"
        "If you use a similar query to the one you have used before, you will receive the same information. If you are going to call the retrieval tool again, you need to change the approach of the query to retrieve different information.\n\n"

        "# CHAT HISTORY:\n"
        "Here you have the chat history with the previous reasoning and steps made:\n"
        "{chat_history}\n\n"

        "# TOOL CALLING:\n"  
        "To call the tool for retrieve information you need to declare this arguments: (ALL (5) MUST BE DECLARED)\n"
        "{tool_instructions}.\n"
        "If you use the tool more than once, you need to change the argument 'query' to retrieve different information.\n\n"
              
        "# FINAL ANSWER CONSIDERATIONS:\n"
        "When formulating your final answer based on the retrieved guideline content, you must follow these safety and reasoning principles:\n\n"
        
        "1. Base your answer strictly on the information retrieved from the clinical guidelines. Do not rely on assumptions, prior knowledge, or unstated clinical intuition.\n"
        "2. Provide a clear and well-reasoned explanation that helps the physician understand the logic behind the answer.\n"
        "3. If the retrieved content is ambiguous, limited, or does not directly address the question, clearly state this in your answer. Explain what is known and what remains uncertain.\n"
        "4. ⚠️ Be extremely cautious when interpreting low clinical probability or apparent mild presentation of disease. DO NOT conclude that a diagnostic test or treatment is unnecessary based only on an assumption that the patient is “unlikely” to have pulmonary embolism (PE), or because symptoms seem “not severe”.\n"
        "5. Always prioritize the following:\n"
        "   - Justification rooted in guideline content\n"
        "   - Safe and cautious clinical communication\n"
        "   - Explicit mention of limitations, risks, and need for follow-up when appropriate\n\n"
    
        "Always prioritize caution, transparency, and alignment with the clinical source material.\n\n" 
                      
        "# OUTPUT FORMAT:\n"
        "In the first steps, the response must be a ToolCall message with the tool call to retrieve the information you need or detailed and reasoned response with the answer to the question.\n"
        "Your final mission is to generate a detailed and reasoned answer to the user query, based on the retrieved information.\n"
        "When all need information is retrieved, you must generate a final response to the user query.\n\n"
        
        "Follow this output format for the final response:\n"
        "The response will be delivered to a physician; use a professional, clear, and well-organized style.\n"
        "No introductions or meta-comments, do not start with “The response to the query…” or mention tool calls.\n"
        "- Focus exclusively on the asked question; omit information that does not contribute to the case.\n"
        "For straightforward questions provide a brief direct answer. It can be 1-2 sentences.\n"
        "For complex questions including clinical cases, provide a reasoned response, including reasoning steps if appropriate.\n"
        "- Be precise: avoid vague words or redundancies.\n"
        "Conclude with a very brief summary of the key points, if necessary.\n"
    )
    
    #Variables Retrieval Tool
    prev_retrieval_queries = state["retrieval_queries"]
    retrieved_docs = state["retrieved_information"]
    
    # Configure remaining of tool calls:
    remaining_tool_calls = 2 - len(prev_retrieval_queries)
    
    # Configure retrieval queries
    retrieval_queries = [f"Query {i+1}: {query}" for i, query in enumerate(prev_retrieval_queries)]
    retrieval_queries = "\n".join(retrieval_queries)
    
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
    
    tool_instructions = retrieval_tool.args
    
    # Format the prompt
    llm_input = prompt.format(
        user_query=user_query,
        retrieved_info= retrieved_information,
        chat_history= state["external_messages"],
        tool_instructions=tool_instructions,
        remaining_tool_calls=remaining_tool_calls,
        retrieval_queries=retrieval_queries,
    )

    # Invoke the LLM
    retrieval_llm = llm.bind_tools([retrieval_tool])
    response = retrieval_llm.invoke(llm_input)

    print(f"### LLM Response: {response} ###")

    # Add response to state
    state["messages"].append(response)
    # Save as external message when not a tool_calls
    if not response.tool_calls:
        
        # Print the response content
        print(f"The answer to the user query: {user_query} is:\n {response.content}\n")
        
        # Extract hallucinations
        hallucination_detector = HallucinationDetectorFactory.load_detector("RefChecker")
        detected_hallucinations = hallucination_detector.detect_hallucinations(response.content, docs)

        print(f"###### Detected Hallucinations: ######\n")
        print(detected_hallucinations)
        
        # Create a GuidelinesConsultationResponse message
        query_response_message = GuidelinesConsultationResponse(
            content=response.content,
            sources=retrieved_docs,
            query=user_query,
        )
        state["external_messages"].append(query_response_message)
        state["messages"].append(query_response_message)
        state["guidelines_consultation_responses"].append(query_response_message)
        
    return state