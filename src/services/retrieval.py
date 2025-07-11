#################################
##### ---    IMPORTS    --- #####
#################################
from langchain.schema import Document
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

from src.llm_config import *
from src.services.re_ranking import *

#############################################
##### ---    CLASSES & FUNCTIONS    --- #####
#############################################

##### ---   Vectorstore   --- #####

persist_directory = "vectorstores/pe_protocol/"

vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"), #text-embedding-ada-002
    collection_name="pe_protocol",
    persist_directory=persist_directory
)

##### ---   Re-Ranking   --- #####

def rerank(query: str, docs: List[Document], method: str = 'Listwise', top_k: int = 4, *qwars) -> List[Document]:
    """ Rerank the documents retrieved by the retriever using the specified reranking method. 

    Args:
        query (str): The query provided to the user to search for documents.
        docs (List[Document]): The documents retrieved by the retriever.
        method (str): The reranking method to use.
        top_k (int): The number of documents to return after reranking.

    Returns:
        List[Document]: The reranked documents.
    """
    reranker = Reranker_Factory.load_reranker(method_name=method, top_n=top_k)
    reranked_docs = reranker.rerank_docs(query, docs)
    return reranked_docs



##### ---   Reference Retrieval   --- #####

def second_retrieval(retrieved_docs: list[Document]) -> list[Document]:
    """
    Enrich retrieved documents with referenced figures, tables, and sections.

    Args:
        retrieved_docs (list): List of documents retrieved in the first search.
        vectorstore (Chroma): The vector database to query.

    Returns:
        list: Original documents enriched with referenced chunks.
    """
    # Extract all references from the retrieved documents
    all_references = set()
    for doc in retrieved_docs:
        if 'References' in doc.metadata:
            references = doc.metadata['References'].split(", ")
            all_references.update(references)

    if all_references == set():
        return retrieved_docs
    
    # Separate references into types
    figure_table_refs = [ref for ref in all_references if ref.startswith(("Figure", "Table", "Supplementary Table"))]

    # Track already retrieved document IDs to avoid duplicates
    retrieved_doc_ids = {doc.metadata.get("id") for doc in retrieved_docs if "id" in doc.metadata}

    # Retrieve referenced figures and tables
    referenced_chunks = []
    for ref in figure_table_refs:
        # Create a filter dict to match the reference
        filter_dict = {"Figure/Table/SupplementaryTable": {"$eq": ref}}
        results = vectorstore.similarity_search(
            query="",  # Empty query since we're filtering by metadata
            filter=filter_dict  # Use the filter dictionary
        )

        # Add only documents not already retrieved
        for result in results:
            if result.metadata.get("id") not in retrieved_doc_ids:
                referenced_chunks.append(result)
                retrieved_doc_ids.add(result.metadata.get("id"))

    # Combine initial documents with referenced chunks
    enriched_docs = retrieved_docs + referenced_chunks

    return enriched_docs


##### ---   Retrieval   --- #####

def retrieve(query: str, top_k: int = 12,reranking: bool = True, top_k_rerank: int = 3, include_refs: bool = False) -> Tuple[str, list[Document]]:
    """ Retrieve documents from the vectorstore based on a query.

    Args:
        query (str): The query to search for in the vectorstore.
        top_k (int, optional): The number of documents to retrieve. Defaults to 12.
        include_refs (bool, optional): Whether to include referenced figures, tables, and sections. Defaults to False.

    Returns:
        serialized_result (str): String with all the content of all retreived documents.
        final_results (list): List of all retrieved documents.
    """
    initial_results = vectorstore.similarity_search(query=query, k=top_k)
    
    if reranking == True:
        reranked_results = rerank(query = query, docs = initial_results, method = "Listwise", top_k = top_k_rerank)
        initial_results = reranked_results
    
    if include_refs==True:
        final_results = second_retrieval(initial_results)
    else:
        final_results = initial_results

    serialized_results = "\n\n".join(
        f"Content: {doc.page_content}\n" for doc in final_results
    )
    return serialized_results, final_results