# step 1 : setup llm (mistral with huggingFave )
import os
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA



# step 1 : setup llm (mistral with huggingFace)
def load_llm():
    embedding_model = HuggingFaceEmbeddings(
    model_name="local_model",#local_model --> sentence-transformers/all-MiniLM-L6-v2
    encode_kwargs={"normalize_embeddings": True}
    )
    return embedding_model 



# step 2: create a custom_prompt_template.

CUSTOM_PROMPT_TEMPLATE  = ''' Use the pieces of information provided in the context to answer the user's question. 
If the answer is not present in the context, respond with "I don’t know" — do not fabricate or assume information. 
Do not provide anything outside of the given context. 

By reviewing the question and the context, generate the following structured topics:

1. Information of disease  
2. Disease precaution  
3. Disease medicine to cure it  
4. Home remedies to ease pain or cure it  

Format the response clearly under these headings.  
Start the answer directly, without any introductory or closing remarks.  

Context: {context}  
Question: {question}  
'''



def set_custom_prompt(CUSTOM_PROMPT_TEMPLATE):
    prompt  = PromptTemplate(template = CUSTOM_PROMPT_TEMPLATE, input_variables = ['context', 'question'])
    return prompt



# embeddingd_model = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")

DB_FAISS_PATH = 'vectorstore/db_faiss'
# embedding_model= SentenceTransformer("local_model",device ='cpu')
llm = OllamaLLM(model="tinyllama") # language model can also use llama3 



db =FAISS.load_local(DB_FAISS_PATH, load_llm() , allow_dangerous_deserialization = True)



# step 3:  Create Qa-chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type ='stuff',
    retriever = db.as_retriever(search_kwargs= {'k':5}),
    return_source_documents = True,
    chain_type_kwargs = {'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)


# now invoke with a single query ....
user_query = input(" Write query Here : ")
response =qa_chain.invoke({'query':user_query})



print("Result:", response['result'])
print("\n------*100\n\n")
print("Source Document : ",response['source_documents'])

