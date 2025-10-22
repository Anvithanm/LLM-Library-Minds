import nbformat

path = "RAG-llm-knowledge-integration/RAG_Book_Based_LLM.ipynb"  # change this to your file name

nb = nbformat.read(path, as_version=4)
if "widgets" in nb.metadata:
    del nb.metadata["widgets"]
nbformat.write(nb, open(path, "w", encoding="utf-8"))
print("Cleaned widget metadata — re-upload to GitHub.")
