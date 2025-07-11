#################################
##### ---    IMPORTS    --- #####
#################################

from langchain_core.messages import AIMessage, SystemMessage

from src.custom_config.state_schema import CustomState

from src.graph_compilation import *

from src.llm_config import *

#############################
##### ---    APP    --- #####
#############################

state = CustomState(
    external_messages=[AIMessage(content="Welcome! How can I assist you today?")],
    messages=[SystemMessage(content="Excel path: data/clinical_cases/clinical_cases.xlsx")],
    patient_data={},
    metrics={},
    clinical_case_evaluation_reports=[],
    guidelines_consultation_responses=[],
    retrieved_information=[],
    retrieval_queries=[],
    finish=False,
    prepare_final_message=False,
)

# Specify an ID for the thread
config = {"configurable": {"thread_id": "ab"}, "recursion_limit": 100}

# Compile the graph
graph = compile_graph()

# Save the graph
visualize_graph(graph, save_name="graph")

# Initialize the streaming process
stream = graph.stream(state, config=config, stream_mode="values")

finish = False
# Iterate through the events
while not finish:
    try:
        event = next(stream)

        if event["finish"]:
            print("Session Finished.")
            finish = True
    
    except Exception as e:
        print(f"Error: {e}")
        break

## DISPLAY RESULTS
# Display external messages
print("\n#####################################\n")
print("- CHAT HISTORY:\n")
chat_history = event["external_messages"]
for message in chat_history:
    message.pretty_print()
print("\n#####################################\n")

# Display the internal messages
print("\n#####################################\n")
print("- INTERNAL MESSAGES:\n")
chat_history = event["messages"]
for message in chat_history:
    message.pretty_print()
print("\n#####################################\n")

# Display the patient data
if event['patient_data']:
    print("\n#####################################\n")
    print("- PATIENT DATA:")
    print(event['patient_data'])
    print("\n#####################################\n")

# Display the processed retrieval queries
if event['retrieval_queries']:
    print("\n#####################################\n")
    print("- PROCESSED RETRIEVAL QUERIES:")
    print(event['retrieval_queries'])
    print("\n#####################################\n")

# Display the retrieved information
if event['retrieved_information']:
    print("\n#####################################\n")
    print("- RETRIEVED INFORMATION:")
    print(event['retrieved_information'])
    print("\n#####################################\n")

# Display the Guidelines Consultation Responses
if event['guidelines_consultation_responses']:
    print("\n#####################################\n")
    print('- GUIDELINES CONSULTATION RESPONSE:\n')
    print(f'Query: {event["guidelines_consultation_responses"][0].query}\n')
    print(f'Response: {event["guidelines_consultation_responses"][0].content}\n')
    print(f'Sources: {event["guidelines_consultation_responses"][0].sources}\n')
    print(f'Hallucinations: {event["guidelines_consultation_responses"][0].hallucinations}')
    print("\n#####################################\n")

# Display the Clinical Case Evaluation Reports
if event['clinical_case_evaluation_reports']:
    # Metrics
    print("\n#####################################\n")
    print("- PATIENT CLINICAL METRICS:")
    print(event['metrics'])
    print("\n#####################################\n")

    # Report
    print("\n#####################################\n")
    print(' - CLINICAL CASE EVALUATION REPORT:\n')
    print(f'Query: {event["clinical_case_evaluation_reports"][0].query}\n')
    print(f'Response: {event["clinical_case_evaluation_reports"][0].content}\n')
    print(f'Sources: {event["clinical_case_evaluation_reports"][0].sources}\n')
    print(f'Hallucinations: {event["clinical_case_evaluation_reports"][0].hallucinations}')
    print("\n#####################################\n")