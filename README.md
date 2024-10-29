## What is LangChain?

Langchain is a framework designed to help developers build applications powered by large language models (LLMs) in a structured and efficient way. It simplifies the development of LLM-based tools by providing a variety of components that one can combine to handle different tasks such as document retrieval, text processing, or building conversational agents. Langchain allows seamless integration of LLMs into workflows, such as generating text, summarizing information, answering questions, or interacting with external APIs.

### What is loaders?

**Loaders** are components used to extract data or documents from various sources and make them available for the retrieval system. 

Loaders deal with the specifies of accessing and converting data.

- Accessing:
    - Websites
    - Databases
    - You tube
    - arXiv etc.
- Data Types:
    - PDF
    - HTML
    - JSON
    - Word
    - Power point etc.
- Return a list of ‘Document’ Object

# Document Splitting:

Document splitting is a crucial preprocessing step in LangChain that divides large texts into smaller, manageable chunks for efficient processing by language models. Proper splitting ensures better context preservation and more accurate responses.

## Types of Splitters

LangChain provides several text splitter classes:

## **1.** `CharacterTextSplitter`:

- Simplest splitter that divides text based on a specified character count
- Splits on a single separator
- Useful for basic text splitting needs.

Use Case:

1. Simple text documents
2. Performance-critical applications
3. Texts with consistent structure

## 2. `RecursiveCharacterTextSplitter`

- Sophisticated version of the `CharacterTextSplitter`
- Most versatile and commonly recommended text splitter
- Tries to split based on different characters in a recursive manner.
- First, it will split on more prominent characters like paragraphs, then sentences, and finally individual characters.

Use Case:

1. Documents with clear paragraph structures
2. Mixed content types
3. When semantic coherence is crucial

### 3. MarkdownHeaderTextSplitter

- Specialized for Markdown documents, splits based on headers.
- Maintains document hierarchy
- Preserves header metadata
- Enables hierarchical navigation

 Use Case:

1. Wiki pages
2. Markdown-based content
3. Technical documentation
4. Documentation splitting

### 4. TokenTextSplitter

- Splits text based on token count rather than character count
- Handles special tokens
- Uses the model’s tokenizer to split the text by the number of tokens, ensuring compatibility with token limits of language models.

Use Case:

1. LLM-specific applications
2. When token count is critical
3. API cost optimization
4. Complex language processing

### 5. PythonCodeTextSplitter

- Optimized for splitting Python source code.
- Respects Python syntax
- Maintains code block integrity
- Preserves function and class boundaries

Use Case:

1. Python source code
2. Code documentation
3. API documentation
4. Code analysis tasks

### 6. SpacyTextSplitter

- Uses spacy for linguistics-aware splitting
- Multi-language support
- Sentence boundary detection
- Named entity preservation

Use Case:

1. Natural language processing
2. Text analysis tasks
3. Multi-language documents
4. When linguistic accuracy is crucial

### 7. NLTKTextSplitter

- Uses NLTK for natural language-aware splitting
- Uses NLTK's sentence tokenizer
- Supports multiple languages with NLTK models
- Preserves sentence boundaries

Use Case:

1. Academic text analysis
2. Linguistic research
3. Multi-language document processing
4. When sentence boundary detection is crucial

### 8. Language()

- A specialized splitter for programming language source code
- Maintains code block integrity
- Language-specific syntax awareness
- Handles multiple programming languages

Use Case:

1. Source code analysis
2. Code documentation generation
3. API documentation

### 9. SentenceTransformersTokenTextSplitter

- Uses sentence transformers for semantic-aware splitting
- Uses transformer models for tokenization
- Supports multiple languages through transformer models
- Maintains semantic coherence

Use Case:

1. Document summarization
2. Semantic analysis
3. Content clustering
4. Question-answering systems
5. When semantic coherence is crucial

# **Vectorstores and Embeddings**

In LangChain, **vectorstores** and **embeddings** are essential components that work together to enable retrieval-augmented generation (RAG), document search, and other memory-based tasks for large language models (LLMs).

## Vectorstores

Vectorstores are databases that store and index these vector embeddings, enabling. Once text is converted into vectors (embeddings), these vectors are stored in a **vectorstore**

- Efficient similarity search
- Nearest neighbor lookups
- Semantic retrieval operations

## **Embeddings**

**Embeddings** are numerical representations of text or any data in a high-dimensional space. These vector representations allow us to measure the similarity between texts by comparing the distances between their vectors. In LangChain, embeddings are used to convert text data into vectors, enabling efficient semantic search.

LangChain, embeddings convert text into vectors that can be:

- Compared for similarity
- Stored in vector databases
- Used for semantic search

# Retrieval:

Retrieval is the centerpiece of our retrieval augmented generation (RAG) flow.

Retrieval refers to the process of finding relevant pieces of information from a collection of data (such as documents, PDFs, or any unstructured text) based on a user's query. This is an important step in building systems that interact with large datasets and provide meaningful responses, particularly for tasks like question-answering, summarization, and chatbots.

To implement retrieval in LangChain follow these key steps:

1. Data Loading
2. Splitting the document
3. Embedding the document
4. Vector Stores (Storing the Embedding)
5. Retrieving Relevant Documents

## What is MMR algorithm?

The **MMR (Maximal Marginal Relevance)** algorithm is used in information retrieval and natural language processing to improve the diversity of results in search or summarization tasks. Its primary goal is to balance **relevance** (how closely a document or sentence matches the query) with **novelty** (how different the document or sentence is from the other already retrieved items).
## What is LLM aided retrieval?

**LLM-aided retrieval** refers to the use of **Large Language Models (LLMs)** to enhance the process of information retrieval. Instead of solely relying on traditional retrieval techniques that depend on keyword matching or vector-based similarity search, LLMs can assist by interpreting complex queries, reasoning over data, and generating more context-aware retrieval strategies. This improves the overall quality and relevance of the retrieved information.

## What is Compression?

**Compression** refers to reducing the size of a dataset, document, or information while preserving the most relevant or important content. Compression helps improve efficiency and ensures that only essential information is passed through different stages of processing, like retrieval, summarization, or response generation.

Use FAISS in stead of ChromaDB here:

- No SQLite dependency (which was causing issues)
- Better performance for similarity search
- Simpler setup
- Works well with large datasets
