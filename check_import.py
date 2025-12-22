try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("Import from langchain_text_splitters successful")
except ImportError as e:
    print(f"Import from langchain_text_splitters failed: {e}")
