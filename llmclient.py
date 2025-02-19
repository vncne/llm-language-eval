# llmclient.py

from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import requests
import google.generativeai as genai

class LLMClient:
    def __init__(self, provider: str, api_key: str = None):
        self.provider = provider.lower()
        self.api_key = api_key

        if self.provider not in ["openai", "anthropic", "deepseek", "perplexity", "google"]:
            raise ValueError(f"Unsupported provider: {self.provider}")
        if not api_key:
            raise ValueError(f"No API key provided for {self.provider}.")

        if self.provider == "google":
            genai.configure(api_key=self.api_key)

    def chat(self, model: str, system_prompt: str, user_prompt: str, **overrides) -> str:
        if self.provider == "openai" and "o1-mini" in model.lower():
            raise ValueError(f'Unsupported model: "o1-mini"')
        else:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
        
        if self.provider == "openai":
            llm = ChatOpenAI(
                model_name=model,
                openai_api_key=self.api_key,
                **overrides
            )
        elif self.provider == "anthropic":
            llm = ChatAnthropic(
                model_name=model,
                anthropic_api_key=self.api_key,
                **overrides
            )
        elif self.provider == "deepseek":
            llm = ChatOpenAI(
                model_name=model,
                openai_api_key=self.api_key,
                openai_api_base="https://api.deepseek.com/v1",
                **overrides
            )
        elif self.provider == "perplexity":
            llm = ChatPerplexity(
                model_name=model,
                perplexity_api_key=self.api_key,
                **overrides
            )
        elif self.provider == "google":
            return self._chat_google(model, system_prompt, user_prompt, **overrides)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        response = llm.invoke(messages)
        if isinstance(response, dict):
            return response.get("content", "").strip()
        return response.content.strip()

    def _chat_google(self, model: str, system_prompt: str, user_prompt: str, **overrides) -> str:
        generation_config = {
            "temperature": overrides.get("temperature", 0.9),
            "top_p": overrides.get("top_p", 1.0),
            "top_k": overrides.get("top_k", 1),
            "max_output_tokens": overrides.get("max_tokens", 2048),
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]

        model = genai.GenerativeModel(model_name=model,
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        chat = model.start_chat(history=[])
        response = chat.send_message(f"{system_prompt}\n\n{user_prompt}")
        return response.text





class ChatPerplexity:
    def __init__(self, model_name: str, perplexity_api_key: str, **overrides):
        self.model_name = model_name
        self.perplexity_api_key = perplexity_api_key
        self.overrides = overrides

    def invoke(self, messages):
        # Build the payload with the system and user messages.
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": messages[0].content},
                {"role": "user", "content": messages[1].content}
            ]
        }
        # Set default parameters for Perplexity; these can be overridden via overrides.
        defaults = {
            "temperature": 0.2,
            "top_p": 0.9,
            "search_domain_filter": ["perplexity.ai"],
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": "month",
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1,
            "response_format": None,
            "max_tokens": 2000,

        }
        # Merge any user-provided overrides
        defaults.update(self.overrides)
        payload.update(defaults)

        headers = {
            "Authorization": f"Bearer {self.perplexity_api_key}",
            "Content-Type": "application/json"
        }
        url = "https://api.perplexity.ai/chat/completions"
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        # data = response.json()

        data = response.json()
        # Extract the actual text response:
        content = data["choices"][0]["message"]["content"].strip()
        # Return an object with a 'content' attribute to mimic other LLMs.
        return type("Response", (), {"content": content})



        # return response["choices"][0].message.content.strip()
