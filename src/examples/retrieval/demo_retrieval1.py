
import src.configs.config  
from loguru import logger
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



logger.info("============== prepare data =================")
src_doc = "src/examples/retrieval/docs/src_golden_hymns_of_epictetus.txt"
output_doc = "src/examples/retrieval/docs/output_golden_hymns_of_epictetus_new.txt"

start_saving = False
stop_saving = False
line_to_save =[]

with open(src_doc, "r", encoding="utf-8") as f:
     for i, line in enumerate(f.readlines()):
         if i > 2000: break # Limit lines for speed demo
         if "Are these the only works of Providence within us?" in line:
             start_saving = True
         if "*** END OF THE PROJECT GUTENBERG EBOOK THE GOLDEN SAYINGS OF EPICTETUS" in line:
             stop_saving = True
         if start_saving and not stop_saving:
             line_to_save.append(line)


# Write the lines to a new file
logger.info("len of line_to_save:" + str(len(line_to_save)))

with open(output_doc, "w", encoding="utf-8") as f:
    f.writelines(line_to_save)

wordcount = 0
with open(output_doc, "r", encoding="utf-8") as f:
    for line in f.readlines():
        wordcount += len(line.split())

logger.info(f"wordcount: {wordcount}")


logger.info("============== prepare data done =================")

logger.info("======= load  text data to langchain============")

loader = TextLoader(file_path=output_doc)
golden_saying_content = loader.load()

logger.info("type of golden_saying_content: " + str(type(golden_saying_content)))  # it's a list
logger.info("len of golden_saying_content: " + str(len(golden_saying_content))) # 1
logger.info("type of golden_saying_conten's first element:" + str(type(golden_saying_content[0]))) # <class 'langchain_core.documents.base.Document'>
logger.info("======= load  text data to langchain done============")



logger.info("========== chunking =============================")         

text_splitter =  RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap =50,
    length_function = len,
    add_start_index = True # start_index will be added to metadata of each chunk
)
texts = text_splitter.split_documents(golden_saying_content)
logger.info(texts[0]) # include page_content, metadata(source, start_index)
logger.info(texts[1])

logger.info("========== chunking done =============================")         

# Add dummy metadata for filtering demonstration
for i, text in enumerate(texts):
    if i < 2:
        text.metadata['chapter'] = 'introduction'
    else:
        text.metadata['chapter'] = 'main_content'

logger.info("Added 'chapter' metadata to chunks for demo.")


logger.info("==========  text embedding =============================")
import os
logger.info(f"Checking GOOGLE_API_KEY: {bool(os.getenv('GOOGLE_API_KEY'))}")

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# FAISS (Facebook AI Similarity Search): 
# Efficient similarity search and clustering of dense vectors. Optimized for speed/memory, runs locally.
# need  pip install faiss-cpu or faiss-gpu

# Note on other Vector Store options:
# 1. Google Cloud Platform (GCP): You can use "Vertex AI Vector Search" (formerly Matching Engine).
#    Class: langchain_google_vertexai.VectorSearch
# 2. PostgreSQL on Cloud (e.g., Cloud SQL): Use the "pgvector" extension.
#    Library: langchain-postgres
#    Class: langchain_postgres.PGVector

# # 使用 Google Gemini Embeddings
# 需要确保环境变量中有 GOOGLE_API_KEY
vector_store = FAISS.from_documents(documents=texts, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")) # need to setup env GOOGLE_API_KEY = <your gemini api key>

# vector_store = FAISS.from_documents(documents=texts, embedding=OpenAIEmbeddings())# we donot use this as we don't hav the openai api key

logger.info("==========  embedding done =============================")  

logger.info("============= query from vector store =====================")
str_query =" how can I practice mindfulness if I am always busy and distracted" # 如果我总是很忙碌、很容易分心，我该如何练习正念（Mindfulness）呢？

# k=10 means retrieve the top 10 most similar document chunks (Top-K). default is 20
# The result is a list of Document objects (containing text content and metadata), NOT a direct answer.
# Note: 'p' (Top-P) is usually for LLM generation sampling, not for vector retrieval.

# Other FAISS search methods available:
# 1. similarity_search_with_score(query, k): Returns list of (Document, score) tuples.
# 2. max_marginal_relevance_search(query, k, fetch_k): Optimize for similarity + diversity (MMR).
# 3. similarity_search_by_vector(embedding, k): Search using raw embedding vector.
rs = vector_store.similarity_search(query=str_query, k=10)
logger.info("type of rs:" + str(type(rs)) + " len of rs: " + str(len(rs)))
for r in rs:
    logger.info(r)


logger.info("============= query with filter =====================")
# Example of using filter to restrict results to a specific 'chapter' metadata
# Note: LangChain's FAISS implementation supports basic metadata filtering via a dictionary.
# We use a query relevant to the introduction and increase fetch_k to ensure candidates are found.
str_query_intro = "works of Providence"
rs_filtered = vector_store.similarity_search(
    query=str_query_intro, 
    k=10, 
    filter={'chapter': 'introduction'}, # Only matches documents where metadata has chapter='introduction'
    fetch_k=100
)
logger.info("type of rs_filtered:" + str(type(rs_filtered)) + " len of rs_filtered: " + str(len(rs_filtered)))
for r in rs_filtered:
    logger.info(f"Filtered result metadata: {r.metadata}")
logger.info("============= query from vector done=====================")




logger.info("============= query with llm =============================")
# Use LCEL (LangChain Expression Language) instead of RetrievalQA (which requires 'langchain' package)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.llm.gemini_chat_model import get_gemini_llm

# 1. Retrieve with Scores
# We manually retrieve to get access to similarity scores
docs_and_scores = vector_store.similarity_search_with_score(str_query, k=5)

# Format context with Source ID and Score
context_parts = []
for i, (doc, score) in enumerate(docs_and_scores):
    # Note: FAISS returns distance (lower is better for L2), or similarity (higher is better for Cosine)
    # Depending on FAISS index type. Assuming L2 distance here for standard FAISS.
    # If using Cosine, score is similarity.
    # Let's label it "Score/Distance".
    context_parts.append(f"[Source {i+1}] (Score: {score:.4f}):\n{doc.page_content}")

formatted_context = "\n\n".join(context_parts)

# 2. Setup Prompt with Citation Instructions
template = """Answer the question based only on the following context. 
Please cite the sources you used for your answer (e.g., [Source 1], [Source 2]).
At the end of your answer, list the sources with their scores.

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 3. Setup LLM
llm = get_gemini_llm()

# 4. Build Chain (Simple Prompt -> LLM -> Parser)
chain = prompt | llm | StrOutputParser()

# 5. Invoke Chain
response = chain.invoke({"context": formatted_context, "question": str_query})
logger.info(f"LLM Response:\n{response}")

# Optional: Print raw top sources for debugging
logger.info("--- Top Retrieval Results ---")
for i, (doc, score) in enumerate(docs_and_scores):
    logger.info(f"[Source {i+1}] Score: {score:.4f}, Content snippet: {doc.page_content[:50]}...")

logger.info("done")
