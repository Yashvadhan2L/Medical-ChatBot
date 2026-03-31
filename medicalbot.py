import streamlit as st 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate

CUSTOM_PROMPT_TEMPLATE = """
 You are a highly reliable medical assistant. Your task is to answer strictly based on the provided context.
 STRICT RULES:
 - Use ONLY the information present in the given context.
 - DO NOT add external knowledge, assumptions, or general medical advice.
 - If the answer is not clearly found in the context, respond ONLY with:
   "I don’t know"
 - DO NOT hallucinate or infer missing details.
 - DO NOT include disclaimers, introductions, or conclusions.
 - DO NOT mention the word "context" in your answer.
 INSTRUCTIONS:
 Carefully analyze both the question and the context. Extract only relevant information and organize it into the following structured format.
 OUTPUT FORMAT (STRICTLY FOLLOW):
 1. Disease Information:
 - (Clear explanation of the disease)
 2. Disease Precautions:
 - (List all precautions mentioned in the context)
 3. Medicines / Treatment:
 - (Only medicines or treatments explicitly mentioned)
 4. Home Remedies:
 - (Only remedies explicitly mentioned)
 IMPORTANT:
 - If ANY section is not present in the context, write: "Not available"
 - Keep answers concise and relevant
 - Do NOT merge sections
 - Do NOT add extra sections
 Context:
 {context}
 Question:
 {question}
"""     

def load_llm():
    llm =  OllamaLLM(model="tinyllama" , temperature = 0.9)
    return llm

def set_custom_prompt(CUSTOM_PROMPT_TEMPLATE):
    prompt  = PromptTemplate(template = CUSTOM_PROMPT_TEMPLATE, input_variables = ['context', 'question'])
    return prompt

DB_FAISS_PATH = 'vectorstore/db_faiss'
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="local_model") 
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization = True)
    return db

def set_custom_prompt():
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=['context', 'question']
    )
    return prompt


def main():
    st.title("Medical ChatBot ")
    st.write("i am a medical chatbot , How can i help you ")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here .. ")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user','content': prompt})
        
    try :
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("failed to load the vectorstore ")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm = load_llm(),
            chain_type ='stuff',
            retriever = vectorstore.as_retriever(search_kwargs= {'k':6}),
            return_source_documents = True,
            chain_type_kwargs = {'prompt':set_custom_prompt()}
        )

        response  = qa_chain.invoke({'query':prompt})
        result = response['result']
        source_documents = response['source_documents']
        # result_to_show = result + str(source_documents)

        st.chat_message('assistant').markdown(result)
        st.session_state.messages.append({'role': 'assistant','content': result})

    except Exception as e:
        st.error(f'Error {str(e)}')



if __name__ == "__main__":
    main()
