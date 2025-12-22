
from langchain_community.vectorstores import FAISS
import inspect

print(inspect.signature(FAISS.similarity_search))
