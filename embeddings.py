import logging
from typing import List, Dict, Optional, Sequence
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class DocumentEmbeddings:
    def __init__(self, collection_name: str = "doc_embeddings"):
        self.logger = logging.getLogger(__name__)
        self.collection_name = collection_name
        self.dim = 384  # Dimension of all-MiniLM-L6-v2 embeddings
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Connect to Milvus Lite
        try:
            connections.connect(alias="default", uri="milvus_lite.db")
            self.logger.info("Connected to Milvus Lite")
            self._init_collection()
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus Lite: {str(e)}")
            raise

    def _init_collection(self):
        """Initialize Milvus collection with required schema"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields=fields, description="Documentation embeddings")
        
        # Create collection if it doesn't exist
        if self.collection_name not in utility.list_collections():
            self.collection = Collection(name=self.collection_name, schema=schema)
            # Create index for vector similarity search
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            self.logger.info(f"Created collection: {self.collection_name}")
        else:
            self.collection = Collection(name=self.collection_name)
            self.logger.info(f"Using existing collection: {self.collection_name}")
        
        self.collection.load()

    def _chunk_text(self, text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling - take attention mask into account for correct averaging"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings for a list of texts"""
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return self._mean_pooling(model_output, encoded_input['attention_mask']).numpy()

    def add_documents(self, urls: Sequence[str], contents: Sequence[str], batch_size: int = 100) -> bool:
        """Add multiple documents to the vector store in batches"""
        try:
            total_docs = len(urls)
            if total_docs != len(contents):
                raise ValueError("Number of URLs must match number of contents")

            for i in range(0, total_docs, batch_size):
                batch_urls = urls[i:i + batch_size]
                batch_contents = contents[i:i + batch_size]
                
                # Split contents into chunks
                all_chunks = []
                all_urls = []
                for url, content in zip(batch_urls, batch_contents):
                    chunks = self._chunk_text(content)
                    all_chunks.extend(chunks)
                    all_urls.extend([url] * len(chunks))
                
                # Generate embeddings for all chunks
                embeddings = self._get_embeddings(all_chunks)
                
                # Prepare data for insertion
                data = [
                    all_urls,
                    all_chunks,
                    embeddings.tolist()
                ]
                
                # Insert batch
                self.collection.insert(data)
                self.logger.info(f"Added batch of {len(all_chunks)} chunks from {len(batch_urls)} documents")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents batch: {str(e)}")
            return False

    def add_document(self, url: str, content: str) -> bool:
        """Add a single document to the vector store"""
        return self.add_documents([url], [content], batch_size=1)

    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """Search for similar documents using the query"""
        try:
            # Generate query embedding
            query_embedding = self._get_embeddings([query])
            
            # Search in Milvus
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query_embedding[0].tolist()],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["url", "content"]
            )
            
            # Format results and deduplicate by URL
            similar_docs = []
            seen_urls = set()
            for hits in results:
                for hit in hits:
                    url = hit.entity.get("url")
                    if url not in seen_urls:
                        similar_docs.append({
                            "url": url,
                            "content": hit.entity.get("content"),
                            "score": hit.score
                        })
                        seen_urls.add(url)
            
            return similar_docs
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return []

    def clear(self):
        """Clear all documents from the collection"""
        try:
            self.collection.drop()
            self._init_collection()
            self.logger.info("Cleared all documents")
        except Exception as e:
            self.logger.error(f"Failed to clear documents: {str(e)}")
