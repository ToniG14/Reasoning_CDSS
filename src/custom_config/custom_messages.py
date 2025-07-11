#################################
##### ---    IMPORTS    --- #####
#################################
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, AIMessage
from typing import Optional, List, Dict, Tuple

#############################################
##### ---    CLASSES & FUNCTIONS    --- #####
#############################################

##### ---   Parameter Request   --- #####

class ParameterRequestMessage(AIMessage):
    """Custom message to request specific patient parameters."""
    def __init__(self, parameters: List[str], user_input: Optional[Dict[str, str]] = None):
        super().__init__(
            content=(
                f"The following parameters are missing and require user input: {', '.join(parameters)}."
                " Please provide values for each."
            )
        )
        self.parameters = parameters  # Store the list of parameters being requested
        self.user_input = user_input or {}  # Add a dictionary for user-provided inputs


##### ---   User Query   --- #####

class UserQuery(HumanMessage):
    """Custom message type to represent a user query."""
    def __init__(self, content: str):
        """
        Initialize a UserQuery message.

        Args:
            content (str): The content of the user's query.
        """
        super().__init__(content=content)


##### ---   GuidelinesConsultationResponse   --- #####

class GuidelinesConsultationResponse(AIMessage):
    """Custom message to store a response from a guidelines consultation."""
    def __init__(self, content: str, sources: Optional[List[Document]] = None, query: Optional[str] = None, hallucinations: Optional[List[Tuple[str, str]]] = None):
        """
        Args:
            content (str): The response content.
            sources (str): Sources or context retrieved and used to generate the response.
            query (str): The user query that led to this response.
            hallucinations (List[Tuple[str,str]]): List of hallucinations and their scores.
        """
        super().__init__(content=content)
        self.sources = sources
        self.query = query
        self.hallucinations = hallucinations


##### ---   ClinicalCaseEvaluationReport   --- #####

class ClinicalCaseEvaluationReport(AIMessage):
    """Custom message to store a clinical case evaluation report."""
    def __init__(self, content: str, sources: Optional[List[Document]] = None, query: Optional[str] = None, hallucinations: Optional[List[Tuple[str,str]]] = None):
        """
        Args:
            content (str): The text of the clinical case evaluation report.
            sources (str): Sources or context used to generate the report.
            query (str): The user query that led to this report.
            hallucinations (List[Tuple[str,str]]): List of hallucinations and their scores.
        """
        super().__init__(content=content)
        self.sources = sources
        self.query = query
        self.hallucinations = hallucinations

##### ---   Next Action   --- #####

class Action(AIMessage):
    "Custom message to store the action to be taken by the system elaborated by the planner."
    def __init__(self, content: str):
        """
        Args:
            content (str): Description of the action to be taken.
        """
        super().__init__(content=content)
        

##### ---   Tool Request   --- #####

class ToolRequest(AIMessage):
    """Custom message to store a response calling a tool. The response must contain the args for the Tool."""
    def __init__(self, content: str, args: Optional[Dict] = None):
        """
        Args:
            content (str): The complete response to the user's query.
            args (dict): Dict with the args names and its values.
        """
        super().__init__(content=content)
        self.args = args


##### ---   Finish Eval   --- #####

class FinishEval(AIMessage):
    """
    Custom message to indicate whether the evaluation should continue or finish.
    """
    def __init__(self, content: str):
        """
        Args:
            content (str): 'continue' to proceed with another recommendation or 'finish' to end the session.
        """
        super().__init__(content=content)

