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
    api_key = "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"  # Replace with your actual API key or environment variable
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

            def invoke(self, prompt: str, use_streaming=True) -> str:
                try:
                    if use_streaming:
                        # Handle streaming response
                        stream = self.client.messages.create(
                            model=self.model,
                            max_tokens=64000,
                            temperature=0.5,
                            system=self.system_prompt,
                            stream=True,
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )

                        # Collect content from the stream
                        full_response = ""
                        for chunk in stream:
                            if chunk.type == "content_block_delta" and hasattr(chunk, "delta") and hasattr(chunk.delta,
                                                                                                           "text"):
                                full_response += chunk.delta.text

                        return full_response.strip()
                    else:
                        # Handle non-streaming response
                        response = self.client.messages.create(
                            model=self.model,
                            max_tokens=64000,
                            temperature=0.5,
                            system=self.system_prompt,
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )
                        return response.content[0].text.strip()
                except Exception as e:
                    logging.error(f"Claude API call failed: {e}")
                    return f"Error generating response: {e}"

            def predict(self, prompt: str) -> str:
                # By default, use non-streaming for predict method to match your original code's behavior
                return self.invoke(prompt, use_streaming=False)

        print(f"Attempting to connect to Claude API with model '{ClaudeClient(api_key).model}'...")
        client = ClaudeClient(api_key)
        return client

    except ImportError:
        logging.error("Anthropic package not found. Install with: pip install anthropic")
        print("Falling back to placeholder LLM.")
        return get_fallback_llm()
    except Exception as e:
        logging.error(f"An unexpected error occurred setting up Claude client: {e}")
        print("Falling back to placeholder LLM.")
        return get_fallback_llm()
