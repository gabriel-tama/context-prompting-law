import os
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, GPTVectorStoreIndex
from llama_index.response.pprint_utils import pprint_response
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings

from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
import torch
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.llms import HuggingFaceLLM
from llama_index import ListIndex

from typing import Any, List
from InstructorEmbedding import INSTRUCTOR
from llama_index.embeddings.base import BaseEmbedding



from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)
import torch
from llama_index import (
    GPTVectorStoreIndex,
    LangchainEmbedding,
    LLMPredictor,
    ServiceContext,
    StorageContext,
    download_loader,
    PromptHelper,
    VectorStoreIndex,
)


# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")


# NOTE: the first run of this will download/cache the weights, ~20GB
hf_predictor = HuggingFaceLLM(
    context_window=8192,
    max_new_tokens=1024,
    generate_kwargs={"temperature": 0, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="TheBloke/wizardLM-13B-1.0-fp16",
#     model_name="lmsys/vicuna-13b-v1.3",
    model_name="TheBloke/wizardLM-13B-1.0-fp16",
    device_map='auto',
    tokenizer_kwargs={"max_length": 4096},
    tokenizer_outputs_to_remove=["token_type_ids"],

    model_kwargs={"torch_dtype": torch.bfloat16}
)

class InstructorEmbeddings(BaseEmbedding):
    def __init__(
    self, 
    instructor_model_name: str = "hkunlp/instructor-large",
        instruction: str = "Represent the document for pasal retrieval",
    **kwargs: Any,) -> None:
        self._model = INSTRUCTOR(instructor_model_name)
        self._instruction = instruction
        
        super().__init__(**kwargs)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, query]])
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, text]])
        return embeddings[0] 

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode([[self._instruction, text] for text in texts])
        return embeddings

    


service_context = ServiceContext.from_defaults(chunk_size=512, llm=hf_predictor, embed_model=InstructorEmbeddings(embed_batch_size=2))




def save_to_txt(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        print("Data has been successfully saved to", file_path)
    except IOError:
        print("Error: Unable to save data to", file_path)

# Example usage:


for file_name in os.listdir('../Investasi/'):
    doc_1 = SimpleDirectoryReader(input_files=['../Investasi/'+file_name]).load_data()
    doc_1_idx = ListIndex.from_documents(doc_1,service_context=service_context)
    query_engine = doc_1_idx.as_query_engine(service_context=service_context)
    response = query_engine.query("Is the document correlated?")
    data_to_save = response.response
    file_path = './result/'+file_name+"_correlated.txt"
    save_to_txt(file_path, data_to_save)
    response = query_engine.query("Is the document has direct contradiction?")
    data_to_save = response.response
    file_path = './result/'+file_name+"_contradiction.txt"
    save_to_txt(file_path, data_to_save)
