import os
import logging
import anthropic

def get_fallback_llm():
    """Fallback placeholder LLM when Claude isn't available"""
    class PlaceholderLLM:
        def invoke(self, prompt: str) -> str:
            print(f"--- PLACEHOLDER LLM PROMPT ---\n{prompt}\n--- END PROMPT ---")
            if "update description for entity" in prompt.lower():
                return "Placeholder updated entity description."
            elif "analyze the python function named" in prompt.lower() and "output format exactly like this" in prompt.lower():
                return """Origin: Unknown (Placeholder)
Description: Placeholder description (Claude unavailable)."""
            return "Placeholder LLM response (Claude unavailable)"

        def predict(self, prompt: str) -> str:
            return self.invoke(prompt)

    print("Using Placeholder LLM.")
    return PlaceholderLLM()


def get_llm_client():
    """Returns a Claude API client configured for code understanding tasks."""
    api_key = "ANTHROPIC_API_KEY"  # Replace with your actual API key or environment variable
    if not api_key:
        logging.error("Claude API key not found in ANTHROPIC_API_KEY environment variable.")
        print("Falling back to placeholder LLM.")
        return get_fallback_llm()

    try:
        from anthropic import Anthropic

        class ClaudeClient:
            def __init__(self, api_key: str):
                self.client = Anthropic(api_key=api_key)
                self.model = "claude-3-7-sonnet-20250219"
                self.system_prompt = (
                    "You are an AI specialized in code analysis and documentation. "
                    "Provide concise, accurate responses about code structure and behavior. "
                    "When asked for a description, provide ONLY the description text."
                )

            def invoke(self, prompt: str) -> str:
                try:
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=512,
                        temperature=0.2,
                        system=self.system_prompt,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    # Extract the text from the response's content
                    return response.content[0].text.strip()
                except Exception as e:
                    logging.error(f"Claude API call failed: {e}")
                    return f"Error generating response: {e}"

            def predict(self, prompt: str) -> str:
                return self.invoke(prompt)
        print(f"Attempting to connect to Claude API with model '{ClaudeClient(api_key).model}'...")
        client = ClaudeClient(api_key)
        # Optionally, you could perform a test call here to verify connectivity
        return client

    except ImportError:
        logging.error("Anthropic package not found. Install with: pip install anthropic")
        print("Falling back to placeholder LLM.")
        return get_fallback_llm()
    except Exception as e:
        logging.error(f"An unexpected error occurred setting up Claude client: {e}")
        print("Falling back to placeholder LLM.")
        return get_fallback_llm()
