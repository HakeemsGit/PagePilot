import gradio as gr
import logging
from typing import Tuple, Iterator
from scraper import DocumentationScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def process_url(url: str, progress=gr.Progress()) -> Iterator[Tuple[str, str, str]]:
    """
    Process a documentation URL for ingestion
    Returns status message, discovered URLs, and any errors
    """
    try:
        scraper = DocumentationScraper()
        progress(0, desc="Starting URL discovery...")
        
        urls = scraper.discover_urls(url)
        total_urls = len(urls)
        
        progress(0.5, desc=f"Found {total_urls} documentation pages")
        
        # For now, just return the discovered URLs
        urls_text = "\n".join(urls)
        progress(1.0, desc="Completed URL discovery")
        
        yield f"Successfully processed {url}", urls_text, ""
        
    except Exception as e:
        yield "", "", f"Error processing URL: {str(e)}"

def query_docs(question: str) -> Tuple[str, str]:
    """
    Query the processed documentation
    Returns answer and source links
    """
    # TODO: Implement actual RAG querying
    return f"Answer to: {question}", "Source: example.com"

# Create the Gradio interface
with gr.Blocks(title="Documentation Assistant") as demo:
    gr.Markdown("# Documentation Assistant")
    
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
            outputs=[status_output, urls_output, error_output]
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
