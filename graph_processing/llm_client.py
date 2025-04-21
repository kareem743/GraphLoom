import logging

def get_fallback_llm():
    """Fallback placeholder LLM when Ollama isn't available"""
    class PlaceholderLLM:
        def invoke(self, prompt: str) -> str:
            print(f"--- PLACEHOLDER LLM PROMPT ---\n{prompt}\n--- END PROMPT ---")
            if "update description for entity" in prompt.lower():
                return "Placeholder updated entity description."
            elif "analyze the python function named" in prompt.lower() and "output format exactly like this" in prompt.lower():
                return """Origin: Unknown (Placeholder)
Description: Placeholder description (Ollama unavailable)."""
            return "Placeholder LLM response (Ollama unavailable)"
        def predict(self, prompt: str) -> str: return self.invoke(prompt)
    print("Using Placeholder LLM.")
    return PlaceholderLLM()

def get_llm_client():
    """Returns an Ollama LLM client configured for code understanding tasks."""
    try:
        from ollama import Client

        class OllamaClient:
            def __init__(self):
                self.client = Client(host='http://localhost:11434')
                self.model = 'llama3.2'
                self.system_prompt = (
                    "You are an AI specialized in code analysis and documentation. "
                    "Provide concise, accurate responses about code structure and behavior. "
                    "When asked for a description, provide ONLY the description text."
                )

            def invoke(self, prompt: str) -> str:
                try:
                    response = self.client.generate(
                        model=self.model,
                        system=self.system_prompt,
                        prompt=prompt,
                        options={'temperature': 0.2}
                    )
                    return response.get('response', '').strip()
                except Exception as e:
                    logging.error(f"Ollama API call failed: {e}")
                    try:
                        self.client.list()
                    except Exception as conn_e:
                        logging.error(f"Ollama connection test failed: {conn_e}")
                        logging.error("Please ensure Ollama server is running and accessible at the specified host.")
                    return f"Error generating response: {e}"

            def predict(self, prompt: str) -> str:
                return self.invoke(prompt)

        print(f"Attempting to connect to Ollama and verify model '{OllamaClient().model}'...")
        test_client = OllamaClient()
        try:
            model_info = test_client.client.show(test_client.model)
            if not model_info:
                raise ConnectionError(f"Model '{test_client.model}' not found or Ollama inaccessible.")
            print(f"Ollama model '{test_client.model}' verified.")
            print("Ollama connection successful.")
            return test_client
        except Exception as e:
            print(f"Ollama connection/verification failed: {e}")
            print(f"Falling back to placeholder LLM. Make sure Ollama is running (`ollama serve`) and the model '{test_client.model}' is available (`ollama pull {test_client.model}`).")
            return get_fallback_llm()

    except ImportError:
        print("Ollama Python package not found. Install with: pip install ollama")
        print("Falling back to placeholder LLM.")
        return get_fallback_llm()
    except Exception as e:
        print(f"An unexpected error occurred setting up Ollama: {e}")
        print("Falling back to placeholder LLM.")
        return get_fallback_llm()