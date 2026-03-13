import json
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from pathlib import Path

VLM_OUTPUT = "vlm_descriptions.json"
CHROMA_DIR = "chroma_db"
REPORTS_DIR = "rag_reports"

def build_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def generate_report(description, defect_class, anomaly_score, retriever, llm):
    docs = retriever.invoke(description)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""You are an industrial quality control expert.
A defect has been detected on a glass bottle with the following description:
"{description}"

Anomaly score: {anomaly_score:.4f}
Defect class: {defect_class}

Based on the following knowledge base:
{context}

Write a concise quality control report with:
1. Defect summary
2. Probable causes
3. Corrective actions
4. Urgency level (Low/Medium/High)
"""
    return llm.invoke(prompt)

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    with open(VLM_OUTPUT, "r") as f:
        descriptions = json.load(f)
    
    retriever = build_retriever()
    llm = OllamaLLM(model="llama3")
    
    for item in descriptions:
        path = Path(item["image_path"])
        
        img_name = path.stem
        defect_class = item["defect_class"]

        # prende solo test/contamination/000.png
        short_path = Path(*path.parts[-3:])

        print(f"Processing: {defect_class}/{img_name}")
        
        report = generate_report(
            description=item["description"],
            defect_class=defect_class,
            anomaly_score=item["anomaly_score"],
            retriever=retriever,
            llm=llm
        )
        
        filename = f"{defect_class}_{img_name}.txt"
        filepath = os.path.join(REPORTS_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write(f"Image: {short_path}\n")
            f.write(f"Class: {defect_class}\n")
            f.write(f"Anomaly Score: {item['anomaly_score']:.4f}\n")
            f.write(f"VLM description: {item['description']}\n")
            f.write("="*60 + "\n\n")
            f.write(report)
        
        print(f"  ✓ Saved {filename}")
        

    print(f"\n✓ Reports saved in '{REPORTS_DIR}/'")

if __name__ == "__main__":
    main()
