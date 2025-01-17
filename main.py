import logging
import asyncio
from typing import Tuple
import gradio as gr

from test_data import TEST_URLS
from documentation_scraper import DocumentationScraper
from embeddings import DocumentEmbeddings
from agent import CustomAgent
from llm_config import LLM_CONFIGS, set_api_key, get_api_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def process_url(url: str, progress=gr.Progress()) -> Tuple[str, str, str]:
    """
    Process a documentation URL for ingestion
    Returns status message, discovered URLs, and any errors
    """
    try:
        scraper = DocumentationScraper()
        progress(0, desc="Starting URL discovery...")
        
        # Create event loop if it doesn't exist
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        urls = scraper.discover_urls(url)
        total_urls = len(urls)
        
        if total_urls == 0:
            return "", "", f"No valid URLs found at {url}"
        
        progress(0.5, desc=f"Found {total_urls} documentation pages")
        urls_text = "\n".join(urls)
        progress(1.0, desc="Completed URL discovery")
        
        return f"Successfully processed {url}", urls_text, ""
        
    except Exception as e:
        logging.error(f"Error processing URL: {str(e)}", exc_info=True)
        return "", "", f"Error processing URL: {str(e)}"

def initialize_llm(llm_choice: str) -> CustomAgent:
    """Initialize LLM based on selection"""
    if llm_choice not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM choice: {llm_choice}")
    
    config = LLM_CONFIGS[llm_choice]
    llm = config["class"](**config["kwargs"])
    return CustomAgent(task="documentation_assistant", llm=llm)

# Global variables
doc_llm = None
doc_embeddings = DocumentEmbeddings()
vector_store = None

def update_llm(llm_choice: str, api_key: str = "") -> str:
    """Initialize or update the LLM based on selection and optional API key"""
    global doc_llm
    
    if api_key:
        set_api_key(llm_choice, api_key)
    elif not get_api_key(llm_choice):
        return f"Please enter API key for {LLM_CONFIGS[llm_choice]['name']}"

    try:
        doc_llm = initialize_llm(llm_choice)
        return f"Successfully initialized {LLM_CONFIGS[llm_choice]['name']}"
    except Exception as e:
        return f"Error initializing LLM: {str(e)}"

def query_docs(question: str) -> Tuple[str, str]:
    """Query the documentation with a question"""
    if not vector_store:
        return "Please process documentation first", ""
        
    # Get relevant documents
    docs = vector_store.similarity_search(question, k=3)
    
    # Format response
    answer = "Based on the documentation:\n\n"
    sources = set()
    
    for doc in docs:
        answer += f"{doc.page_content}\n\n"
        sources.add(doc.metadata["source"])
    
    sources_text = "Sources:\n" + "\n".join(sources)
    
    return answer, sources_text

# Create the Gradio interface
with gr.Blocks(title="Documentation Assistant") as demo:
    gr.Markdown("# Documentation Assistant")
    
    # LLM Selection
    with gr.Row():
        llm_choice = gr.Dropdown(
            choices=list(LLM_CONFIGS.keys()),
            value="openai",
            label="Select LLM Provider",
            info="Note: Claude models use the Anthropic API key"
        )
        api_key_input = gr.Textbox(
            label="API Key (optional)",
            type="password",
            placeholder="Enter API key if not set in environment"
        )
    
    llm_status = gr.Textbox(label="LLM Status", interactive=False)
    
    llm_choice.change(
        fn=update_llm,
        inputs=[llm_choice, api_key_input],
        outputs=[llm_status]
    )
    api_key_input.change(
        fn=update_llm,
        inputs=[llm_choice, api_key_input],
        outputs=[llm_status]
    )
    
    with gr.Tab("Add Documentation"):
        url_input = gr.Textbox(
            label="Documentation URL",
            placeholder="Enter base documentation URL (e.g., https://docs.example.com/)"
        )
        process_button = gr.Button("Process Documentation")
        status_output = gr.Textbox(label="Status")
        urls_output = gr.TextArea(label="Discovered URLs", max_lines=10)
        error_output = gr.Textbox(label="Errors")
        
        process_button.click(
            fn=process_url,
            inputs=[url_input],
            outputs=[status_output, urls_output, error_output],
            show_progress=True
        )
    
    with gr.Tab("Query Documentation"):
        question_input = gr.Textbox(label="Your Question")
        query_button = gr.Button("Ask")
        answer_output = gr.Textbox(label="Answer")
        sources_output = gr.Textbox(label="Sources")
        
        query_button.click(
            fn=query_docs,
            inputs=[question_input],
            outputs=[answer_output, sources_output]
        )

if __name__ == "__main__":
    demo.launch()
