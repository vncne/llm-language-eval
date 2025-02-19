# File: main.py
from llmclient import LLMClient
from embedding_scorer import EmbeddingScorer
from evaluation_metrics import EvaluationMetrics
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def run_benchmark(target_language, llm_configs, api_keys, test_data, embedding_api_key):
    console = Console()
    console.print(
        Panel(
            f"[bold blue]LLM Translation Benchmarking[/bold blue]\nTarget Language: [bold]{target_language}[/bold]",
            expand=False
        )
    )
    
    # Initialize LLM clients using provided configurations
    llm_clients = {}
    for config in llm_configs:
        provider = config["provider"].lower()
        key = api_keys.get(provider)
        if not key:
            console.print(f"[red]No API key provided for {provider}![/red]")
            continue
        try:
            llm_clients[config["name"]] = {
                "client": LLMClient(provider=provider, api_key=key),
                "model": config["model"],
                "cosine_similarities": []
            }
        except Exception as e:
            console.print(f"[red]Error initializing {config['name']}:[/red] {e}")
            continue

    # Translation function
    def translate(client: LLMClient, text: str, model: str) -> str:
        system_prompt = (
            f"You are a helpful translator. Translate the user input from English to {target_language}. "
            "Do not include any introductory or concluding words or any commentary. Output only the translation directly."
        )
        user_prompt = text
        return client.chat(model=model, system_prompt=system_prompt, user_prompt=user_prompt)

    # Initialize evaluators
    embedding_scorer = EmbeddingScorer(api_key=embedding_api_key, embedding_model="text-embedding-ada-002")
    evaluator = EvaluationMetrics()

    # Process each test case
    for source_text, correct_answer in test_data.items():
        console.print(f"\n[bold underline]Source:[/bold underline] {source_text}")
        console.print(f"[bold underline]Correct Translation:[/bold underline] {correct_answer}")
        
        table = Table(title="LLM Results", show_lines=True)
        table.add_column("LLM", style="magenta", no_wrap=True)
        table.add_column("Translation", style="green")
        table.add_column("Similarity", justify="right", style="yellow")
        table.add_column("BLEU Score", justify="right", style="cyan")
        table.add_column("ROUGE-L F1", justify="right", style="blue")

        for llm_name, info in llm_clients.items():
            client = info["client"]
            model = info["model"]
            try:
                predicted_answer = translate(client, source_text, model)
                cosine_sim = embedding_scorer.compute_similarity(predicted_answer, correct_answer)
                bleu = evaluator.bleu_score(correct_answer, predicted_answer)
                rouge_dict = evaluator.rouge_scores(correct_answer, predicted_answer)
                rouge_l = rouge_dict['rouge-l']['f']  # get ROUGE-L F1 score
                info["cosine_similarities"].append(cosine_sim)
            except Exception as e:
                predicted_answer = f"Error: {e}"
                cosine_sim = 0.0
                bleu = 0.0
                rouge_l = 0.0
                info["cosine_similarities"].append(cosine_sim)
            
            table.add_row(
                llm_name,
                predicted_answer[:100] + " ...",
                f"{cosine_sim:.4f}",
                f"{bleu:.4f}",
                f"{rouge_l:.4f}"
            )
        
        console.print(table)
    
    # Print summary of cosine similarities
    summary_table = Table(title="Average Scores", show_lines=True)
    summary_table.add_column("LLM", style="magenta", no_wrap=True)
    summary_table.add_column("Final Score", justify="right", style="yellow")
    
    for llm_name, info in llm_clients.items():
        avg_similarity = (
            sum(info["cosine_similarities"]) / len(info["cosine_similarities"])
            if info["cosine_similarities"] else 0.0
        )
        summary_table.add_row(llm_name, f"{avg_similarity:.4f}")
    
    console.print("\n")
    console.print(summary_table)

if __name__ == "__main__":
    # If run directly, we prompt users to run benchmark.py instead.
    print("Please run benchmark.py for configuration and evaluation.")