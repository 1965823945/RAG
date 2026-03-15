"""Generate static PDFs for the RAG demo."""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

PDF_DIR = Path("docs/pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)


def create_pdf(filename: str, title: str, content: str):
    """Create a PDF file with title and content."""
    c = canvas.Canvas(str(PDF_DIR / filename), pagesize=letter)
    
    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 750, title)
    
    # Content
    c.setFont("Helvetica", 12)
    y = 700
    words = content.split()
    line = ""
    
    for word in words:
        if len(line + word) < 80:
            line += word + " "
        else:
            c.drawString(50, y, line.strip())
            y -= 20
            if y < 50:
                c.showPage()
                c.setFont("Helvetica", 12)
                y = 750
            line = word + " "
    
    if line:
        c.drawString(50, y, line.strip())
    
    c.save()
    print(f"Created: {filename}")


# Create 10 static PDFs with actual content
pdfs = [
    ("rag_introduction.pdf", "Introduction to RAG",
     "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by incorporating external knowledge. Instead of relying solely on training data, RAG systems can retrieve relevant information from a database and use it to generate more accurate and contextually relevant responses. This approach helps overcome limitations of static models, such as outdated knowledge and hallucination. RAG combines the power of retrieval systems with generative AI to provide up-to-date and fact-based answers."),
    
    ("langchain_basics.pdf", "LangChain Basics",
     "LangChain is an open-source framework designed for building applications with large language models. It provides a standardized interface for connecting LLMs to other data sources and computation. Key components include chains for sequencing operations, agents for dynamic tool use, memory for conversation context, and prompts for input formatting. LangChain supports integration with various LLM providers including OpenAI, Anthropic, Hugging Face, and local models. The framework simplifies complex NLP tasks like question answering, summarization, and chatbots."),
    
    ("vector_databases.pdf", "Vector Databases",
     "Vector databases are specialized database systems designed to store and query high-dimensional vector embeddings. Unlike traditional relational databases that handle structured data, vector databases excel at similarity search operations. They can efficiently find the most similar items to a query vector from millions or billions of candidates. Popular vector databases include ChromaDB, Pinecone, Weaviate, Milvus, and FAISS. These databases use approximate nearest neighbor (ANN) algorithms like HNSW and IVF to achieve fast search speeds while maintaining accuracy. Vector databases are essential for semantic search, recommendation systems, and RAG applications."),
    
    ("text_embeddings.pdf", "Text Embeddings",
     "Text embeddings are numerical representations of text that capture semantic meaning in a continuous vector space. Words or phrases with similar meanings are positioned closer together in the embedding space. Popular embedding models include OpenAI's text-embedding-ada-002, sentence-transformers, and open-source options like BGE and MPNet. Embeddings transform variable-length text into fixed-length vectors, typically with hundreds or thousands of dimensions. These representations enable mathematical operations like similarity calculation using cosine similarity or dot product. High-quality embeddings are crucial for retrieval accuracy in RAG systems."),
    
    ("document_chunking.pdf", "Document Chunking",
     "Document chunking is the process of splitting large documents into smaller, manageable pieces for processing. Effective chunking strategies are essential for RAG systems because chunk size affects both retrieval precision and response quality. Common approaches include fixed-size chunks with overlap, semantic splitting by headings or paragraphs, and recursive chunking that progressively divides text. The chunk_size parameter typically ranges from 200 to 2000 tokens, while overlap helps maintain context between chunks. Choosing the right chunking strategy depends on the document structure and use case."),
    
    ("semantic_search.pdf", "Semantic Search",
     "Semantic search goes beyond keyword matching to understand the meaning behind queries. Instead of looking for exact word matches, semantic search uses embeddings to find documents with similar meaning to the query. This approach handles synonyms, paraphrases, and conceptual relationships that traditional keyword search cannot capture. Semantic search works by converting both queries and documents into vector embeddings, then finding the nearest matches in the vector space. It significantly improves user experience by returning more relevant results, even when the exact search terms don't appear in the documents. Semantic search is fundamental to RAG's retrieval component."),
    
    ("prompt_engineering.pdf", "Prompt Engineering",
     "Prompt engineering is the practice of crafting effective inputs to large language models to achieve desired outputs. It involves carefully designing the prompt structure, including instructions, context, and examples. Key techniques include zero-shot prompting without examples, few-shot prompting with examples, chain-of-thought reasoning, and role-playing. Effective prompts specify the desired format, contain relevant context, and clearly state the task. In RAG systems, prompt engineering involves structuring the retrieved context and question together. Good prompts can significantly improve the accuracy and relevance of LLM responses."),
    
    ("llm_finetuning.pdf", "Fine-tuning LLMs",
     "Fine-tuning is the process of adapting a pre-trained language model to a specific task or domain. Instead of training from scratch, fine-tuning starts with a model trained on large corpora and continues training on task-specific data. This approach requires far less data and compute than training from scratch. Common fine-tuning methods include full parameter fine-tuning and parameter-efficient techniques like LoRA and QLoRA. Fine-tuned models can specialize in particular domains like medical, legal, or technical content. However, RAG often provides a more flexible alternative to fine-tuning, as it can incorporate new information without model retraining."),
    
    ("retrieval_systems.pdf", "Retrieval Systems",
     "Retrieval systems find relevant documents from a large collection based on user queries. Modern retrieval systems go beyond simple keyword matching to use semantic understanding. The retrieval process typically involves encoding the query into embeddings, searching a vector database for similar vectors, and ranking results by relevance. Key metrics include precision, recall, and mean reciprocal rank. Advanced techniques include hybrid search combining keyword and semantic approaches, re-ranking models that refine initial results, and query expansion to improve recall. Effective retrieval is the foundation of successful RAG implementations."),
    
    ("generative_ai.pdf", "Generative AI",
     "Generative AI refers to artificial intelligence systems that can create new content including text, images, audio, and video. Large language models are a prominent example, capable of generating human-like text based on patterns learned during training. Modern generative models like GPT-4, Claude, and Gemini use transformer architectures with billions of parameters. These models can write code, answer questions, summarize text, and engage in natural conversations. Generative AI has applications across industries including content creation, customer service, education, and software development. However, challenges remain around accuracy, bias, and ensuring responsible use.")
]

for filename, title, content in pdfs:
    create_pdf(filename, title, content)

print(f"\nCreated {len(pdfs)} static PDFs in {PDF_DIR}")
