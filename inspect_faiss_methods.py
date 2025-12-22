
from langchain_community.vectorstores import FAISS
import inspect

# List all methods of FAISS class that contain 'search'
methods = [m for m in dir(FAISS) if 'search' in m]
print("FAISS search methods:")
for m in methods:
    print(f"- {m}")
