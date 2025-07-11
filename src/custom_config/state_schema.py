#################################
##### ---    IMPORTS    --- #####
#################################
from langgraph.graph import MessagesState
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langchain.schema import Document

from src.custom_config.custom_messages import ClinicalCaseEvaluationReport, GuidelinesConsultationResponse

#############################################
##### ---    CLASSES & FUNCTIONS    --- #####
#############################################

##### ---   Custom State   --- #####

class CustomState(MessagesState):
    messages: Annotated[list[AnyMessage], add_messages]
    external_messages: Annotated[list[AnyMessage], add_messages]
    patient_data: dict = {}
    metrics: dict = {}
    clinical_case_evaluation_reports: list[ClinicalCaseEvaluationReport] = []
    guidelines_consultation_responses: list[GuidelinesConsultationResponse] = []
    finish: bool = False
    retrieved_information: list[list[Document]] = []
    retrieval_queries: list[str]
    prepare_final_message: bool = False
    
