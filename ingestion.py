from IPython.display import Image, display
import os
from langchain_mistralai import MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import base64
from IPython.display import Image, display


load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def create_index_if_not_exists(client, index_name):
    """
    Create an OpenSearch index with proper mapping for vector search if it doesn't exist.

    Args:
        client: OpenSearch client instance
        index_name: Name of the index to create
    """
    # Delete the index if it exists (to ensure proper mapping)
    if client.indices.exists(index=index_name):
        print(
            f"Deleting existing index '{index_name}' to recreate with proper mappings..."
        )
        client.indices.delete(index=index_name)

    # Get dimension from a sample embedding
    from helper import get_embedding

    sample_embedding = get_embedding("Sample text for dimension detection")
    dimension = len(sample_embedding)
    print(f"Using embedding dimension: {dimension}")

    # Define mappings with vector field for embeddings
    mappings = {
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "content_type": {"type": "keyword"},
                "token_count": {"type": "integer"},
                "embedding": {"type": "knn_vector", "dimension": dimension},
                "base64_image": {"type": "binary", "doc_values": False, "index": False},
                "table_html": {"type": "text", "index": False},
                "metadata": {
                    "properties": {
                        "filename": {"type": "keyword"},
                        "caption": {"type": "text"},
                        "image_text": {"type": "text"},
                    }
                },
            }
        },
        "settings": {
            "index": {
                "knn": True,
                "knn.space_type": "cosinesimil",  # Use cosine similarity for embeddings
            }
        },
    }

    try:
        client.indices.create(index=index_name, body=mappings)
        print(f"Created index '{index_name}' with vector search capabilities.")
    except Exception as e:
        print(f"Error creating index: {e}")
        raise


def prepare_chunks_for_ingestion(chunks):
    """
    Prepare chunks for ingestion by adding embeddings and token counts.

    Args:
        chunks: List of chunks to prepare

    Returns:
        List of prepared chunks ready for ingestion
    """
    from helper import get_embedding, get_token_count

    prepared_chunks = []

    for i, chunk in enumerate(chunks):
        try:
            # Skip chunks without content
            if not chunk.get("content"):
                continue

            # Compute embedding
            embedding = get_embedding(chunk["content"])

            # Compute token count
            token_count = get_token_count(chunk["content"])

            # Create document for ingestion
            ingestion_doc = {
                "content": chunk["content"],
                "content_type": chunk.get("content_type", "text"),
                "token_count": token_count,
                "embedding": embedding,
                "metadata": {
                    "filename": chunk.get("filename", ""),
                    "caption": chunk.get("caption", ""),
                    "image_text": chunk.get("image_text", ""),
                },
            }

            # Add image-specific data if available
            if chunk.get("content_type") == "image" and "base64_image" in chunk:
                ingestion_doc["base64_image"] = chunk["base64_image"]

            # Add table-specific data if available
            if chunk.get("content_type") == "table" and "table_as_html" in chunk:
                ingestion_doc["table_html"] = chunk["table_as_html"]

            prepared_chunks.append(ingestion_doc)

            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Prepared {i+1}/{len(chunks)} chunks")

        except Exception as e:
            print(f"Error preparing chunk: {str(e)}")

    print(f"Successfully prepared {len(prepared_chunks)} chunks for ingestion")
    return prepared_chunks


def ingest_chunks_into_opensearch(client, index_name, chunks):
    """
    Ingest prepared chunks into OpenSearch.

    Args:
        client: OpenSearch client instance
        index_name: Name of the index
        chunks: Prepared chunks with embeddings and token counts

    Returns:
        Number of successfully ingested documents
    """
    # Track successful and failed operations
    successful = 0
    failed = 0

    # Use bulk API for better performance
    from opensearchpy.helpers import bulk

    # Prepare bulk operations
    operations = []
    for i, chunk in enumerate(chunks):
        operations.append({"_index": index_name, "_source": chunk})

        # Process in batches of 100
        if (i + 1) % 100 == 0 or i == len(chunks) - 1:
            try:
                success, failed_items = bulk(
                    client, operations, stats_only=True)
                successful += success
                failed += len(operations) - success
                operations = []  # Reset for next batch
                print(f"Ingested {successful} chunks so far ({failed} failed)")
            except Exception as e:
                print(f"Bulk ingestion error: {str(e)}")
                failed += len(operations)
                operations = []  # Reset after error

    # Final bulk operation if any remaining
    if operations:
        try:
            success, failed_items = bulk(client, operations, stats_only=True)
            successful += success
            failed += len(operations) - success
        except Exception as e:
            print(f"Bulk ingestion error: {str(e)}")
            failed += len(operations)

    print(f"Ingestion complete: {successful} successful, {failed} failed")
    return successful


def ingest_all_content_into_opensearch(
    processed_images, processed_tables, semantic_chunks, index_name="localrag"
):
    """
    Process and ingest all content (images, tables, text) into OpenSearch.
    """

    from helper import get_opensearch_client

    # 1. Create OpenSearch client
    client = get_opensearch_client("localhost", 9200)

    # 2. Create index if it doesn't exist
    create_index_if_not_exists(client, index_name)

    # 4. Combine all valid chunks
    all_valid_chunks = processed_images + processed_tables + semantic_chunks
    print(f"Total chunks for ingestion: {len(all_valid_chunks)}")

    # 5. Prepare for ingestion (add embeddings and token counts)
    prepared_chunks = prepare_chunks_for_ingestion(all_valid_chunks)

    # 6. Ingest into OpenSearch
    successful_count = ingest_chunks_into_opensearch(
        client, index_name, prepared_chunks
    )

    return successful_count


if __name__ == "__main__":
    from unstructured.partition.pdf import partition_pdf

    from chunking import (
        create_semantic_chunks,
        process_images_with_captions,
        process_tables_with_descriptions,
    )

    # Process a PDF document end-to-end
    pdf_file_path = "files/2312.10997v5-1-7.pdf"

    # # 1. Extract raw chunks
    raw_chunks = partition_pdf(
        filename=pdf_file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Figure", "Table"],
        extract_image_block_to_payload=True,
        chunking_strategy=None,
    )

    # 2. Process images with captions
    processed_images, image_errors = process_images_with_captions(raw_chunks)
    print(f"Processed {len(processed_images)} images with captions")
    print(f"Image Processing {processed_images[0]}")

    # 3. Process tables with descriptions
    processed_tables, table_errors = process_tables_with_descriptions(
        raw_chunks, use_gemini=True, use_ollama=False
    )
    print(f"Processed {len(processed_tables)} tables with descriptions")
    # print(f"Table Processing {processed_tables[0]}")

    # # 4. Partition the PDF into chunks
    # chunks = partition_pdf(
    #     filename=pdf_file_path,
    #     strategy="hi_res",
    #     chunking_strategy="by_title",
    #     max_characters=2000,
    #     min_chars_to_combine=500,
    #     chars_before_new_chunk=1500,
    # )

    # 5. Create semantic chunks
    # semantic_chunks = create_semantic_chunks(chunks)
    # print(f"Created {len(semantic_chunks)} semantic text chunks")
    # print(f"Semantic Chunk Example: {semantic_chunks[0]}")

    collection_name = "2312"
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = QdrantVectorStore.from_documents(
        documents=processed_images,
        # documents=processed_tables,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name,
    )

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)


prompt = ChatPromptTemplate.from_template("""
You are an expert life insurance advisor assistant.
Use only the provided context to answer clearly and professionally.
If unsure, say: "I don’t have enough information in the documents."

<context>
{context}
</context>

Question: {input}
Answer:
""")

document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# question = "give me SAMPLE ANNUITY ILLUSTRATION"

# response = rag_chain.invoke({"input": question})

# question = "what are the augmentation stages?"
# question = "give me details about Technology tree of RAG research in tree formate"
# question = "explain me representative instance of the RAG process applied to question answering in details with diagram"
# question = "give me Comparison between the three paradigms of RAG."
question = "give me in table SUMMARY OF RAG METHODS"
response = rag_chain.invoke({"input": question})


# for key in response.keys():
#     print(key)


# def display_base64_image(base64_code):
#     # Decode the base64 string to binary
#     image_data = base64.b64decode(base64_code)
#     # Display the image
#     display(Image(data=image_data))


# if 'context' in response:
#     for i, doc in enumerate(response['context']):
#         # print(f"Document {i+1}:")
#         # print(f"  Content preview: {doc.page_content[:100]}...")
#         # print(f"  Metadata: {doc.metadata}")

#         # Check if base64_image exists in metadata or page_content
#         if hasattr(doc, 'metadata') and 'base64_image' in doc.metadata:
#             print(
#                 f"  Has base64_image: Yes (length: {len(doc.metadata['base64_image'])})")
#             # Optionally save or display the image
#             # print(f"  Base64 Image: {doc.metadata['base64_image'][:100]}...")
#             try:
#                 display_base64_image(doc.metadata['base64_image'])
#                 # image_data = base64.b64decode(doc.metadata['base64_image'])
#                 # image = Image.open(io.BytesIO(image_data))
#                 # sixel.sixel_output(image)
#             except Exception as e:
#                 print(f"  Error displaying image: {e}")
#         else:
#             print(f"  Has base64_image: No")
#         print()
# print("\nResponse sKeys:")


print("\n===============================")
print("💬 Final Answer:\n")
print(response["answer"])
print("===============================")

# 6. Ingest all content into OpenSearch

# successful_count = ingest_all_content_into_opensearch(
#     processed_images, processed_tables, semantic_chunks, index_name="localrag"
# )
# print(f"Successfully ingested {successful_count} chunks into OpenSearch")
