api_key = "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"
from anthropic import Anthropic
client = Anthropic(api_key=api_key)
client.messages.create(model='claude-3-7-sonnet-20250219')