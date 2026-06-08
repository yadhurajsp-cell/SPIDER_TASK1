from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the provided context and summarises it ."),
    ("human", "context: {context}\n\nQuestion: {question}")
])



vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

retriever = vectorstore.as_retriever(

    search_type = 'mmr' , search_kwargs = {'k' : 4 ,
    'fetch_k' : 10}
)
llm = ChatMistralAI(model = "mistral-small-2506")

print('your rag system is created ')
print('press 0 to exit the program')



while True:
    query = input("Enter your question: ")
   

    if  query == '0':
        break

    docs = retriever.invoke(query)

    context = "/n/n".join([doc.page_content for doc in docs])


    final_prompt = prompt.invoke({'context': context , 'question' : query} )



    result = llm.invoke(final_prompt)

    print(result.content)

    





