#################################
##### ---    IMPORTS    --- #####
#################################

from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel
from langchain.schema import Document

# CrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# LLMChainFilter
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_openai import ChatOpenAI

# LLMListWise
from langchain.retrievers.document_compressors import LLMListwiseRerank
from langchain_openai import ChatOpenAI

# Embedding
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_openai import OpenAIEmbeddings

# Ensembler
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

# RankGPT
from langchain_core.prompts import PromptTemplate
from pydantic import Field

# RankGPT + Threshold
from statistics import median

# Warning (CrossEncoder)
import warnings

warnings.filterwarnings("ignore", message=r"`clean_up_tokenization_spaces` was not set")
warnings.filterwarnings("ignore", message=r".*Using `tqdm.autonotebook.tqdm` in notebook mode.*")


#############################################
##### ---    CLASSES & FUNCTIONS    --- #####
#############################################

##### ---   ABSTRACT FACTORY CLASS   --- #####

class Reranker(ABC):
    """
    Abstract base class for reranking methods.

    Methods:
    rerank_docs(query, docs):
        Abstract method to rerank the provided documents based on the query.
    """

    @abstractmethod
    def rerank_docs(self, query: str, docs: list[Document]) -> list[Document]:        
        """
        Abstract method to rerank the provided documents based on the query.

        Args:
        query (str): The query used to rerank the documents.
        docs (list): A list of documents to be reranked.
        """
        pass


##### ---   CROSS-ENCODER RERANKER   --- #####

# MiniLM-L12-v2 (Microsoft) is cross-encoder and is very efficient (few layers and parameters, low latency)
# but can fall short for large number of documents
# for large applications you can use ColBERT-v2 (14 Sep 2024) (This is the newest one)
# (jinaai/jina-colbert-v2 (Late-interaction))
# jinaai/jina/jina-reranker-v2-base-multilingual
# You can also use “microsoft/mpnet-base” which is one of the best. There is a v2


class CrossEncoder(Reranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2", top_n: int = 5, **kwargs):
        """
        Initializes the CrossEncoder reranker with the specified model and top_n value.

        Args:
        model_name (str): The name of the pre-trained cross-encoder model to use. Default is "cross-encoder/ms-marco-MiniLM-L-12-v2".
        top_n (int): The number of top documents to return after reranking. Default is 5.
        kwargs: Additional keyword arguments.
        """
        self.model_name = model_name
        self.top_n = top_n
        self.model = HuggingFaceCrossEncoder(model_name=self.model_name)
        self.compressor = CrossEncoderReranker(model=self.model, top_n=self.top_n)

    def rerank_docs(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Reranks the provided documents based on the given query using the cross-encoder model.

        Args:
        query (str): The query string used to rerank the documents.
        docs (List[Document]): A list of documents to be reranked.

        Returns:
        Documents (List[Document]): A sequence of reranked documents.
        """
        # Re-Rank documents
        reranked_docs = self.compressor.compress_documents(docs, query)
        return reranked_docs


##### ---   LLM CHAIN FILTER RERANKER   --- #####
class LLMChainFilterReranker(Reranker):
    def __init__(self, llm:str = 'gpt-4o-mini', **kwargs):
        """
        Initializes the LLMChainFilterReranker with the specified language model.
        Note: This method does not rank the documents but determines which are relevant and which are not.

        Args:
        llm (str): The language model to use for filtering documents.
        kwargs: Additional keyword arguments.
        """
        self.llm = ChatOpenAI(model=llm, temperature=0.0)  # TODO: Factory LLM
        self.compressor = LLMChainFilter.from_llm(self.llm)

    def rerank_docs(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Filters the provided documents based on their relevance to the given query using the LLM chain filter model.

        Note: This method does not rank the documents but determines which are relevant and which are not.

        Args:
        query (str): The query string used to filter the documents.
        docs (List[Document]): A list of documents to be filtered.

        Returns:
        Documents (List[Document]): A sequence of filtered documents.
        """
        # Filter documents using the compressor
        filtered_docs = self.compressor.compress_documents(docs, query)
        return filtered_docs


##### ---   LIST-WISE RERANKER   --- #####
class LLMListwiseReranker(Reranker):
    def __init__(self, llm:str = 'gpt-4o', top_n:int = 5, **kwargs):
        """
        Initializes the LLMListwiseReranker with the specified language model and the top-n ranking threshold.
        This method has proven to be the most effective among those that have been evaluated.

        Args:
        llm (str): The language model instance to use for reranking documents.
        top_n (int): The number of top-ranked documents to return after reranking. Default is 5.
        kwargs: Additional keyword arguments for customization.
        """
        self.llm = ChatOpenAI(model=llm, temperature=0.0)  # TODO: Factory LLM
        self.top_n = top_n
        self.compressor = LLMListwiseRerank.from_llm(self.llm, top_n=self.top_n)

    def rerank_docs(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Reranks the provided documents based on their relevance to the given query using a listwise rerank approach.

        Args:
        query (str): The query string to guide the reranking of documents.
        docs (List[Document]): A list of documents to be reranked.

        Returns:
        Documents (List[Document]): A sequence of reranked documents.
        """
        # Perform listwise reranking using the compressor
        reranked_docs = self.compressor.compress_documents(docs, query)
        return reranked_docs

##### ---   EMBEDDING RERANKER   --- #####


class EmbeddingsReranker(Reranker):
    def __init__(self, similarity_threshold:float = 0.80, top_n: int = 5, **kwargs):
        """
        Initializes the EmbeddingsReranker with a specified similarity threshold and top-n ranking limit.

        Args:
        similarity_threshold (float): The threshold for similarity score between the query and documents to consider them relevant. Default is 0.80.
        top_n (int): The maximum number of documents to return after reranking. Default is 5.
        kwargs: Additional keyword arguments for further customization.
        """
        self.similarity_threshold = similarity_threshold
        self.top_n = top_n
        self.compressor = EmbeddingsFilter(embeddings=OpenAIEmbeddings(), similarity_threshold=self.similarity_threshold, k=self.top_n)

    def rerank_docs(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Reranks the provided documents based on their similarity to the given query using embeddings.

        Args:
        query (str): The query string for similarity-based reranking.
        docs (List[Document]): A list of documents to be reranked.

        Returns:
        Documents (List[Document]): A list of top-n documents that meet the similarity threshold, ordered by relevance.
        """
        # Perform similarity-based re-ranking using the compressor
        reranked_docs = self.compressor.compress_documents(docs, query)
        return reranked_docs

##### ---   ENSEMBLER RERANKER   --- #####

# TODO: This should be configurable in the pipeline,
# TODO: Replace CrossEncoder -> Listwise
# i.e., the Chain Reranker should be instance as yes or no in the pipeline and then decide whether or not to use a reranker of the other types. 
class EnsemblerReranker(Reranker):
    def __init__(self, llm:str = 'gpt-4o-mini', top_n: int = 5, **kwargs):
        """
        Initializes the EnsemblerReranker with both a language model-based filter and a List-wise reranker,
        combining multiple reranking strategies for enhanced relevance filtering.

        Args:
        llm (str): The language model to use in the initial filter phase.
        top_n (int): The maximum number of documents to return after reranking. Default is 5.
        kwargs: Additional keyword arguments for further customization.
        """
        # Params
        self.llm = ChatOpenAI(model=llm, temperature=0.0)  # TODO: Factory LLM  
        self.top_n = top_n

        # Initial filter based on LLM relevance
        self.first_compressor = LLMChainFilter.from_llm(self.llm)

        # Final reranker with top-n constraint
        self.last_compressor = LLMListwiseRerank.from_llm(self.llm, top_n=self.top_n)

        # Combines LLM and cross-encoder reranking steps into a pipeline
        self.pipeline_compressor = DocumentCompressorPipeline(transformers=[self.first_compressor, self.last_compressor])

    def rerank_docs(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Reranks documents based on relevance to the given query by sequentially applying an LLM-based filter
        and a List-wise reranker.

        Args:
        query (str): The query string guiding the reranking process.
        docs (List[Document]): A list of documents to be reranked.

        Returns:
        Documents(List[Document]): A sequence of documents ordered by relevance after applying the ensemble reranking strategy.
        """
        # Apply pipeline compression
        reranked_docs = self.pipeline_compressor.compress_documents(docs, query)
        return reranked_docs
    

##### ---   RANKRGPT RERANKER   --- #####


class RankGPTReranker(Reranker):
    def __init__(self, llm:str = 'gpt-4o-mini', top_n=5, **kwargs):
        """
        Initializes the RankGPTReranker with the specified language model and top-n ranking limit.

        Args:
        llm (str): The language model instance used for generating relevance scores.
        top_n (int): The number of top-ranked documents to return after reranking. Default is 5.
        kwargs: Additional keyword arguments for further customization.
        """
        self.llm = ChatOpenAI(model=llm, temperature=0.0)  # TODO: Factory LLM
        self.top_n = top_n

    # Function that retrieves the reranked docs
    def rerank_docs(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Reranks the documents based on a relevance score from the LLM, using a custom scoring prompt.

        Args:
        query (str): The query string guiding the reranking.
        docs (List[Document]): A list of documents to be reranked.
        top_n (int): The number of top-ranked documents to return. Default is 5.

        Returns:
        List[Document]: A list of top-n documents ordered by relevance to the query.
        """
        # Inner class to represent the structured output of relevance scoring
        class RatingScore(BaseModel):
            relevance_score: float = Field(..., description="The relevance score of a document to a query")

        # Template prompt to elicit relevance rating from the LLM
        prompt_template = PromptTemplate(
            input_variables=["query", "doc"],
            template="""On a scale of 1-10, rate the relevance of the following document to the query.
            Query: {query}
            Document: {doc}
            Relevance Score:"""
        )

        # LLM chain setup for generating structured output with relevance scoring
        llm_chain = prompt_template | self.llm.with_structured_output(RatingScore)

        scored_docs = []
        for doc in docs:
            # Prepares input data for each document with the query
            input_data = {"query": query, "doc": doc.text}
            # Invokes the chain to get relevance score
            try:
                score = llm_chain.invoke(input_data).relevance_score
                score = float(score)  # Ensures score is a float
            except (ValueError, AttributeError):
                score = 0  # Defaults to 0 if scoring fails

            scored_docs.append((doc, score))  # Collects document with its score

        # Sorts documents by relevance score in descending order
        reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

        # Selects the top-n documents and preserves the original structure
        top_docs = [doc for doc, _ in reranked_docs[:(self.top_n)]]

        # Returns the reranked documents in a new RetrievedDocuments object
        return top_docs


##### ---   RANKGPT RERANKER WITH THRESHOLD   --- #####
class RankGPTThresholdReranker(Reranker):
    def __init__(self, llm:str = 'gpt-4o-mini', top_n: int = 5, **kwargs):
        """
        Initializes the RankGPTThresholdReranker with a specified language model and top-n ranking limit.

        Args:
        llm (str): The language model instance used for generating relevance scores.
        top_n (int): The number of top-ranked documents to return after reranking. Default is 5.
        kwargs: Additional keyword arguments for further customization.
        """
        self.llm = ChatOpenAI(model=llm, temperature=0.0)  # TODO: Factory LLM
        self.top_n = top_n

    # Function that retrieves the reranked docs
    def rerank_docs(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Reranks the documents based on relevance scores from the LLM, using a threshold to filter less relevant results.

        Args:
        query (str): The query string guiding the reranking.
        docs (List[Document]): A list of documents to be reranked.
        top_n (int): The number of top-ranked documents to return after filtering. Default is 5.

        Returns:
        List[Document]: A list of documents that exceed the median relevance score, ordered by relevance.
        """
        # Inner class representing the structured output of relevance scoring
        class RatingScore(BaseModel):
            relevance_score: float = Field(..., description="The relevance score of a document to a query")

        # Template prompt to elicit relevance rating from the LLM, emphasizing query intent and context
        prompt_template = PromptTemplate(
            input_variables=["query", "doc"],
            template="""On a scale of 1-10, rate the relevance of the following document to the query. 
            Consider the specific context and intent of the query, not just keyword matches.
            Query: {query}
            Document: {doc}
            Relevance Score:"""
        )

        # LLM chain setup for generating structured relevance scores
        llm_chain = prompt_template | self.llm.with_structured_output(RatingScore)

        # Scoring phase
        scored_docs = []
        for doc in docs:
            input_data = {"query": query, "doc": doc.text}  # Prepare query and document content
            try:
                score = float(llm_chain.invoke(input_data).relevance_score)  # Get relevance score
            except (ValueError, AttributeError):
                score = 0  # Default to 0 if scoring fails
            scored_docs.append((doc, score))

        # Calculate the median threshold
        scores = [score for _, score in scored_docs]
        
        # Check if there are no scores
        if len(scores) == 0:
            return docs # Return the original documents if no scores are available
        
        # Calculate the median threshold
        threshold = median(scores)

        # Filter documents exceeding the median threshold
        filtered_docs = [(doc, score) for doc, score in scored_docs if score > threshold]

        # Sort filtered documents by relevance score
        filtered_docs = sorted(filtered_docs, key=lambda x: x[1], reverse=True)

        # Extract the top_n documents from the filtered list
        top_docs = [doc for doc, _ in filtered_docs[:(self.top_n)]]

        # Return as a RetrievedDocuments object
        return top_docs

##### ---   RERANKER FACTORY CLASS   --- #####

class Reranker_Factory:
    """
    Factory class that instantiates a Reranker based on the configuration.
    """
    def load_reranker(method_name: str, **kwargs) -> Reranker:
        """
        Factory method to instantiate and return the appropriate reranker based on the given method name.

        Args:
            method_name (str): The name of the reranking method to use.
            kwargs (dict): Additional keyword arguments to pass to the reranker class when creating an instance.

        Returns:
            Reranker: An instance of the specified reranker class.

        Supported Rerankers:
            - CrossEncoder: Requires 'model_name' (default: "cross-encoder/ms-marco-MiniLM-L-12-v2"; "jinaai/jina-reranker-v2-base-multilingual"(API) "microsoft/mpnet-base" for best performance), 'top_n' (default: 5).
            - LLM Chain Filter: Requires 'llm'.
            - LLM Listwise: Requires 'llm', 'top_n' (default: 5).
            - Embeddings: Requires 'similarity_threshold' (default: 0.80), 'top_n' (default: 5).
            - Ensembler: Requires 'llm'.
            - RankGPT: Requires 'llm', 'top_n' (default: 5).
            - RankGPT_Threshold: Requires 'llm', 'top_n' (default: 5).

        Raises:
        ValueError: If the method_name is not recognized.
        """
        rerankers = {
            "CrossEncoder": CrossEncoder,
            "Chain": LLMChainFilterReranker,
            "Listwise": LLMListwiseReranker,
            "Embeddings": EmbeddingsReranker,
            "Ensembler": EnsemblerReranker,
            "RankGPT": RankGPTReranker,
            "RankGPT_Threshold": RankGPTThresholdReranker
        }

        if method_name not in rerankers:
            raise ValueError(f"Unknown reranker method: {method_name}, implemented methods are: \n - CrossEncoder \n - Chain \n - Listwise \n - Embeddings \n - Ensembler \n - RankGPT \n - RankGPT_Threshold")

        # Instantiate the appropriate reranker class with provided kwargs
        return rerankers[method_name](**kwargs)

