# Dans le terminal :

# pip install -qU \
#     langchain \
#     openai \
#     pinecone-client\
#     pandas\
#     tiktoken\
#     python-dotenv\
#     streamlit

########################################################################
#               Récupération clés API
########################################################################


from dotenv import load_dotenv
load_dotenv()  # Ceci charge les variables à partir de .env

import os
api_key = os.getenv('PINECONE_API_KEY')  # Utiliser la variable



###############################################################################
#                  Chargement et splitting du fichier excel
#######################################################################################

import pandas as pd
from langchain.schema import Document  # Assurez-vous que le chemin d'importation est correct
from langchain.embeddings.openai import OpenAIEmbeddings

df = pd.read_excel('CRM_done.xlsx')
rows_as_strings = [df.iloc[i].to_string() for i in range(len(df))]

# # Supposons que rows_as_strings est votre liste de chaînes
documents = [Document(page_content=row) for row in rows_as_strings]

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")



#############################################################################
#           Vectorisation et stockage
##############################################################################

from langchain.vectorstores import Pinecone

import pinecone

# initialize pinecone
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),  # find at app.pinecone.io
    environment="gcp-starter",  # next to api key in console
)

# # pinecone.list_indexes()

index_name = "crm"

# First, check if our index already exists. If it doesn't, we create it
if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
# The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
docsearch = Pinecone.from_documents(documents, embeddings, index_name=index_name)

# # if you already have an index, you can load it like this
# docsearch = Pinecone.from_existing_index(index_name="lovebot", embedding=embeddings)


##########################################################
#           Choix du LLM
################################################################


from langchain.chat_models import ChatOpenAI


llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model='gpt-4-vision-preview' #gpt-3.5-turbo-1106
)

###################################################################
#               Retriever et prompt
###################################################################

from langchain.schema.runnable.passthrough import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

retriever=docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 2})

# retriever=docsearch.as_retriever(search_type="mmr",
#     search_kwargs={'k': 5, 'lambda_mult': 1, 'fetch_k':50})

template = """

    Question :
    {question}

    Infos à prendre en compte :

    {context}

    """
rag_prompt_custom = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} | rag_prompt_custom | llm
)
#Je suis account executive dans une entreprise de logiciel B2B. Je vais te donner du contexte (=CRM). Dans ma base de données client qui devrais-je contacter en priorité et pourquoi? Regarde notamment 
question = "Résume-moi les news sur ce site : https://datanews.levif.be/actualite/entreprises/start-ups/la-jeune-pousse-gantoise-introw-leve-1-million-deuros/"

# print(retriever.invoke(question))
reponse = rag_chain.invoke(question)
print(reponse.content)

# print(llm("bonjour").content)
