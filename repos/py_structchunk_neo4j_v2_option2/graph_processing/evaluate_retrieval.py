import logging
from llm_client import get_llm_client

def evaluate_retrieval(
        retrieved_context: dict[str, any],
        query: str,
        llm_client: any,
        metrics: list[str] = ["relevance", "coverage", "diversity"]
) -> None:
    """
    Evaluates the quality of retrieved context against the query and prints the results.

    Args:
        retrieved_context: The context dictionary returned by retrieve_context()
        query: The original user query
        llm_client: An LLM client for evaluation scoring
        metrics: List of metrics to evaluate (relevance, coverage, diversity)
    """
    logging.info("Evaluating retrieval quality...")
    print("\n===== Retrieval Evaluation =====")

    # Print basic stats
    print(f"Query: {query}")
    print(f"Total entities retrieved: {len(retrieved_context.get('top_entities', []))}")
    print(f"Total code chunks retrieved: {len(retrieved_context.get('relevant_chunks', []))}")
    print(f"Total graph nodes: {len(retrieved_context.get('related_graph_nodes', []))}")
    print(f"Total graph relationships: {len(retrieved_context.get('related_graph_relationships', []))}")

    # Skip detailed evaluation if no context was retrieved
    if (not retrieved_context.get("top_entities") and
            not retrieved_context.get("relevant_chunks") and
            not retrieved_context.get("related_graph_nodes")):
        print("Analysis: No context retrieved for evaluation.")
        for metric in metrics:
            print(f"{metric.capitalize()}: 0.0/10")
        return

    # Extract sample content for evaluation
    context_samples = []

    # Add top entities
    for entity in retrieved_context.get("top_entities", [])[:5]:
        meta = entity.get('metadata', {})
        desc = entity.get('description', '')
        entity_type = meta.get('entity_type', 'Unknown')
        name = meta.get('name', 'Unknown')
        context_samples.append(f"{entity_type} '{name}': {desc[:200]}")

    # Add top chunks
    for chunk in retrieved_context.get("relevant_chunks", [])[:5]:
        meta = chunk.get('metadata', {})
        doc = chunk.get('document', '')
        source = meta.get('source_file', 'Unknown')
        context_samples.append(f"Code from {source}: {doc[:200]}")

    # Add node info
    for node in retrieved_context.get("related_graph_nodes", [])[:5]:
        props = node.get('properties', {})
        desc = props.get('description') or props.get('docstring') or ''
        name = props.get('name', 'Unknown')
        context_samples.append(f"Node '{name}': {desc[:200]}")

    # Create evaluation prompt
    evaluation_prompt = f"""
    You are an expert evaluator for code-related information retrieval systems.

    User Query: {query}

    Below are samples from the retrieved context:
    {'---'.join(context_samples)}

    Please evaluate the retrieved context on a scale of 1-10 for the following metrics:
    """

    for metric in metrics:
        if metric == "relevance":
            evaluation_prompt += """
            - Relevance (1-10): How well does the retrieved information relate to the query?
              Score 1: Information is completely irrelevant
              Score 10: Information is highly relevant and directly answers the query
            """
        elif metric == "coverage":
            evaluation_prompt += """
            - Coverage (1-10): How comprehensive is the retrieved information?
              Score 1: Misses key information needed to address the query
              Score 10: Provides complete information needed to address the query
            """
        elif metric == "diversity":
            evaluation_prompt += """
            - Diversity (1-10): How diverse and non-redundant is the retrieved information?
              Score 1: Highly redundant information with many duplicates
              Score 10: Diverse set of relevant information with minimal redundancy
            """

    evaluation_prompt += """
    For each metric, provide only a numerical score (1-10) followed by a one-sentence justification.

    Then provide a brief overall analysis (2-3 sentences) of the retrieved context quality.
    """

    try:
        # Get evaluation from LLM
        evaluation_response = get_llm_client.invoke(evaluation_prompt)
        evaluation_text = evaluation_response.content if hasattr(evaluation_response, 'content') else str(
            evaluation_response)

        # Extract scores using regex
        import re
        print("\nMetrics:")
        for metric in metrics:
            pattern = rf"{metric.capitalize()}\s*\(?1-10\)?\s*:\s*(\d+)"
            match = re.search(pattern, evaluation_text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                print(f"- {metric.capitalize()}: {score}/10")
            else:
                print(f"- {metric.capitalize()}: 0/10 (score not found)")
                logging.warning(f"Could not extract score for {metric}")

        # Extract analysis
        analysis_pattern = r"overall analysis.*?:(.*?)(?=\n\n|\Z)"
        analysis_match = re.search(analysis_pattern, evaluation_text, re.IGNORECASE | re.DOTALL)
        if analysis_match:
            analysis = analysis_match.group(1).strip()
        else:
            # Fallback to extracting the last paragraph
            paragraphs = evaluation_text.split('\n\n')
            analysis = paragraphs[-1].strip()

        print(f"\nAnalysis: {analysis}")

    except Exception as e:
        logging.error(f"Error during retrieval evaluation: {e}")
        print(f"\nEvaluation failed: {str(e)}")


def evaluate_generated_answer(
        query: str,
        retrieved_context: dict[str, any],
        generated_answer: str,
        llm_client: any,
        ground_truth: str = None,
        metrics: list[str] = ["factuality", "completeness", "conciseness", "coherence"]
) -> None:
    """
    Evaluates the quality of the generated answer based on the query and retrieved context
    and prints the results.

    Args:
        query: The original user query
        retrieved_context: The context dictionary returned by retrieve_context()
        generated_answer: The answer generated by the RAG system
        llm_client: An LLM client for evaluation scoring
        ground_truth: Optional ground truth answer for comparison
        metrics: List of metrics to evaluate (factuality, completeness, conciseness, coherence)
    """
    logging.info("Evaluating generated answer quality...")
    print("\n===== Answer Evaluation =====")

    # Print basic info
    print(f"Query: {query}")
    print(f"Answer length: {len(generated_answer)} characters")
    if ground_truth:
        print(f"Ground truth available: Yes")
    else:
        print(f"Ground truth available: No")

    # Create context summary for the evaluation
    context_summary = []

    if retrieved_context.get("top_entities"):
        entity_names = [f"{e.get('metadata', {}).get('name', 'Unknown')}"
                        for e in retrieved_context.get("top_entities", [])[:5]]
        context_summary.append(f"Top entities: {', '.join(entity_names)}")

    if retrieved_context.get("relevant_chunks"):
        file_sources = set([c.get('metadata', {}).get('source_file', 'Unknown')
                            for c in retrieved_context.get("relevant_chunks", [])[:5]])
        context_summary.append(f"Code from files: {', '.join(file_sources)}")

    # Create evaluation prompt
    evaluation_prompt = f"""
    You are an expert evaluator for code-based question answering systems.

    User Query: {query}

    Retrieved Context Summary:
    {' | '.join(context_summary)}

    Generated Answer to Evaluate:
    {generated_answer}
    """

    if ground_truth:
        evaluation_prompt += f"""
        Ground Truth Answer:
        {ground_truth}
        """

    evaluation_prompt += """
    Please evaluate the generated answer on a scale of 1-10 for the following metrics:
    """

    for metric in metrics:
        if metric == "factuality":
            evaluation_prompt += """
            - Factuality (1-10): Does the answer contain only statements that are supported by the retrieved context?
              Score 1: Contains multiple unsupported or incorrect statements
              Score 10: All statements are accurate and supported by the context
            """
        elif metric == "completeness":
            evaluation_prompt += """
            - Completeness (1-10): Does the answer address all aspects of the query?
              Score 1: Ignores major aspects of the query
              Score 10: Comprehensively addresses all aspects of the query
            """
        elif metric == "conciseness":
            evaluation_prompt += """
            - Conciseness (1-10): Is the answer appropriately concise without missing important details?
              Score 1: Excessively verbose or too terse
              Score 10: Optimal length with all necessary information
            """
        elif metric == "coherence":
            evaluation_prompt += """
            - Coherence (1-10): Is the answer well-structured, logical, and easy to follow?
              Score 1: Disorganized, confusing, or difficult to follow
              Score 10: Logically structured, clear, and easy to understand
            """

    evaluation_prompt += """
    For each metric, provide only a numerical score (1-10) followed by a one-sentence justification.

    Then provide:
    1. A brief analysis (2-3 sentences) of the answer's overall quality
    2. Specific suggestions for improvement (2-3 points)
    """

    try:
        # Get evaluation from LLM
        evaluation_response = llm_client.invoke(evaluation_prompt)
        evaluation_text = evaluation_response.content if hasattr(evaluation_response, 'content') else str(
            evaluation_response)

        # Extract scores using regex
        import re
        print("\nMetrics:")
        for metric in metrics:
            pattern = rf"{metric.capitalize()}\s*\(?1-10\)?\s*:\s*(\d+)"
            match = re.search(pattern, evaluation_text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                print(f"- {metric.capitalize()}: {score}/10")
            else:
                print(f"- {metric.capitalize()}: 0/10 (score not found)")
                logging.warning(f"Could not extract score for {metric}")

        # Extract analysis
        analysis_pattern = r"(?:analysis|quality).*?:(.*?)(?=Specific suggestions|\d\.|improvement|$)"
        analysis_match = re.search(analysis_pattern, evaluation_text, re.IGNORECASE | re.DOTALL)
        if analysis_match:
            analysis = analysis_match.group(1).strip()
            print(f"\nAnalysis: {analysis}")

        # Extract improvement suggestions
        suggestions_pattern = r"(?:suggestions|improvement).*?:(.*?)(?=\Z)"
        suggestions_match = re.search(suggestions_pattern, evaluation_text, re.IGNORECASE | re.DOTALL)
        if suggestions_match:
            suggestions = suggestions_match.group(1).strip()
            print(f"\nImprovement Suggestions: {suggestions}")

    except Exception as e:
        logging.error(f"Error during answer evaluation: {e}")
        print(f"\nEvaluation failed: {str(e)}")

    print("\n" + "=" * 30)