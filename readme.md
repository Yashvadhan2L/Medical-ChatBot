# 🩺 Project Synopsis: Medical ChatBot using RAG (FAISS + Ollama)

## Introduction

The Medical ChatBot is an AI-based application developed to provide reliable and context-based medical information. Instead of generating random or general answers, the chatbot is designed to respond only using a predefined set of medical data. This helps in reducing incorrect or misleading information, which is very important in the medical domain.

The project is divided into two main parts: one for creating and storing embeddings, and the other for implementing the chatbot interface and response generation.

---

## Objective

The main aim of this project is to build a medical chatbot that:

* Gives accurate answers based only on available data
* Avoids hallucination or unnecessary assumptions
* Works efficiently using local models
* Provides structured and easy-to-understand responses

---

##  System Overview

The chatbot works using a Retrieval-Augmented Generation (RAG) approach. In simple terms, it first finds relevant information from stored data and then uses a language model to generate the final answer.

---

## Module 1: Embedding and FAISS Storage

In the first part of the project, the focus is on preparing the data:

* Medical documents are collected and divided into smaller parts
* These parts are converted into numerical vectors using a HuggingFace embedding model
* The vectors are then stored using FAISS (a fast similarity search library)
* The data is saved locally in the form of:

  * `index.faiss` (vector data)
  * `index.pkl` (related information/metadata)

This step is important because it allows the chatbot to quickly find relevant information later.

---

## Module 2: Chatbot Implementation

The second part of the project handles user interaction:

* A simple interface is created using Streamlit
* When a user asks a question, it is converted into an embedding
* FAISS is used to find the most relevant information from stored data
* This information is then passed to a language model (TinyLlama using Ollama)
* The model generates a final response based only on the retrieved data

---

##  Models Used

* **Embedding Model**: HuggingFace (for converting text into vectors)
* **Language Model**: TinyLlama via Ollama (for generating responses)

---

##  Prompt Design

A custom prompt template is used to guide the model’s behavior. It ensures that:

* The chatbot only uses the provided context
* No external or assumed information is added
* Answers are structured into sections like:

  * Disease Information
  * Precautions
  * Medicines / Treatment
  * Home Remedies

If the required information is not found, the chatbot simply responds:
"I don’t know"

---

##  Working Process

1. Data is converted into embeddings and stored in FAISS
2. User enters a query
3. The query is converted into an embedding
4. Relevant data is retrieved using FAISS
5. The language model generates a structured answer

---

##  Technologies Used

* Python
* Streamlit
* LangChain
* HuggingFace
* FAISS
* Ollama (TinyLlama)

---

##  Advantages

* Provides accurate and context-based answers
* Reduces chances of incorrect information
* Works locally, ensuring better privacy
* Fast and efficient search using FAISS
* Easy-to-understand structured responses

---

## Limitations

* Depends on the quality and size of the dataset
* Cannot answer beyond stored information
* Initial setup (embedding creation) takes time

---

##  Future Scope

* Use more advanced models like Llama 3 or Mistral
* Deploy as a full web or mobile application
* Add voice-based interaction
* Improve dataset for better coverage

---
##  Conclusion

This project shows how combining embeddings, vector databases, and language models can create a useful and reliable chatbot. By separating the system into two parts—data preparation and chatbot interaction—it becomes more organized and efficient. The use of FAISS makes searching fast, while the language model helps in generating meaningful responses. Overall, the system is a good example of how AI can be used in practical applications like healthcare support.
