from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain.schema import Document

from src.services.retrieval import vectorstore, rerank, second_retrieval


#### --- Input Class --- #####
class RetrievalToolInput(BaseModel):
    query: str = Field(description="The query to search for in the vectorstore. Must be properly designed to retrieve relevant documents.")
    top_k: int = Field(description="The number of documents to retrieve.")
    reranking: bool = Field(default=True, description="Whether to rerank the retrieved documents or not. True by default.")
    top_k_rerank: int = Field(description="Number of documents to keep after reranking.")
    include_refs: bool = Field(description=" Whether to include referenced figures, tables, and sections.")


#### --- Retrieval Tool --- ####
@tool(args_schema=RetrievalToolInput, description="Retrieve documents from the Pulmonary Embolism Guidelines (vectorstore) based on a query.")
def retrieval_tool(query: str, top_k: int = 12, reranking: bool = True, top_k_rerank: int = 3, include_refs: bool = False) -> list[Document]:
    """Retrieve documents from the Pulmonary Embolism Guidelines (vectorstore) based on a query.

    Args:
        query (str): The query to search for in the vectorstore.
        top_k (int): The number of documents to retrieve.
        reranking (bool): Whether to rerank the retrieved documents or not. True by default.
        top_k_rerank (int): Number of documents to keep after reranking.
        include_refs (bool): Whether to include referenced figures, tables, and sections.

    Returns:
        final_results (list): List of all retrieved documents.
    """
    # Print RETRIEVAL TOOL
    print("Retrieval Tool Invoked with the following parameters:\n")
    print(f"Query: {query}\n")
    print(f"Top K: {top_k}\n")
    print(f"Reranking: {reranking}\n")
    print(f"Top K Rerank: {top_k_rerank}\n")
    print(f"Include References: {include_refs}\n")
    
    initial_results = vectorstore.similarity_search(query=query, k=top_k)
  
    if reranking == True:
        reranked_results = rerank(query = query, docs = initial_results, method = "Listwise", top_k = top_k_rerank)
        initial_results = reranked_results
    
    if include_refs==True:
        final_results = second_retrieval(initial_results)
    else:
        final_results = initial_results
  
    metadata = [f"{doc.metadata}\n" for doc in final_results]

    print(f"##### Retrieved Documents #####\n{metadata}\n")
    
    return final_results