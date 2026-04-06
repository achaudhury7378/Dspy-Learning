import dspy
import json
from functions.api import function, String
from functions.sources import get_source
from typing import List, Dict, Any


def call_external_api(source, model: str, messages: List[Dict[str, str]], max_tokens: int = 30000) -> Dict[str, Any]:
    https_connection = source.get_https_connection()
    url = https_connection.url
    client = https_connection.get_client()

    response = client.post(
        url + "/v1/chat/completions",
        json={"model": model, "messages": messages, "temperature": 1.0, "max_completion_tokens": max_tokens},
        timeout=12000,
    )
    return response.json()


class CustomRESTLM(dspy.LM):
    def __init__(self, source_name: str = "Openaiconnector", model: str = "gpt-5-nano", **kwargs):
        max_tokens = kwargs.pop("max_tokens", 20000)
        temperature = kwargs.pop("temperature", 1.0)

        super().__init__(model=model, model_type="chat", temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.source_name = source_name
        self.model_name = model
        self.max_tokens = max_tokens
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def __call__(self, prompt=None, messages=None, **kwargs):
        """Execute the language model call via source integration."""
        if messages is None:
            messages = []
            if hasattr(self, "history"):
                messages.extend(self.history)
            if prompt:
                messages.append({"role": "user", "content": prompt})

        if not hasattr(self, "history"):
            self.history = []

        source = get_source(self.source_name)
        result = call_external_api(source, self.model_name, messages, max_tokens=self.max_tokens)

        self.history.extend(messages if isinstance(messages, list) else [messages])

        if isinstance(result, dict) and "usage" in result:
            usage = result["usage"]
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)

        if isinstance(result, dict):
            text = (
                result.get("choices", [{}])[0].get("message", {}).get("content", "")
                or result.get("output")
                or result.get("text", "")
            )
        else:
            text = str(result)

        return [text]

    def get_usage(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def reset_usage(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0


class SentimentAnalysis(dspy.Signature):
    review: str = dspy.InputField(desc="The movie review text to analyze")
    sentiment: str = dspy.OutputField(desc="The sentiment: positive, negative, or neutral")
    confidence: str = dspy.OutputField(desc="Confidence level: high, medium, or low")


class MovieSentimentAnalyzer(dspy.Module):

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SentimentAnalysis)

    def forward(self, review: str) -> Dict[str, str]:
        prediction = self.predictor(review=review)
        return {"sentiment": prediction.sentiment.strip().lower(), "confidence": prediction.confidence.strip().lower()}


@function(sources=["Openaiconnector"])
def test_source_connection() -> String:
    try:
        source = get_source("Openaiconnector")
        https_connection = source.get_https_connection()
        url = https_connection.url
        return f"Source 'Openaiconnector' found! URL: {url}"
    except Exception as e:
        return f"Error accessing source 'Openaiconnector': {str(e)}"


@function(sources=["Openaiconnector"])
def openai_test(prompt: str) -> String:
    try:
        source = get_source("Openaiconnector")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        result = call_external_api(source, "gpt-5-nano", messages)

        if isinstance(result, dict):
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "") or str(result)
        else:
            content = str(result)

        return content
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"



@function(sources=["Openaiconnector"])
def analyze_movie_sentiment(review: str) -> String:
    try:
        lm = CustomRESTLM(source_name="Openaiconnector", model="gpt-5-nano", temperature=1.0, max_tokens=20000)
        dspy.configure(lm=lm)
        lm.reset_usage()

        analyzer = MovieSentimentAnalyzer()
        result = analyzer.forward(review=review)

        usage = lm.get_usage()

        response = {
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "token_usage": {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            },
        }

        return json.dumps(response, indent=2)
    except Exception as e:
        error_response = {"error": f"{type(e).__name__}: {str(e)}"}
        return json.dumps(error_response, indent=2)


@function(sources=["Openaiconnector"])
def batch_analyze_reviews(reviews: List[str]) -> String:
    try:
        lm = CustomRESTLM(source_name="Openaiconnector", model="gpt-5-nano", temperature=1.0, max_tokens=20000)
        dspy.configure(lm=lm)

        lm.reset_usage()

        analyzer = MovieSentimentAnalyzer()
        results = []

        for review in reviews:
            result = analyzer.forward(review=review)
            results.append(
                {
                    "review": review[:100] + "..." if len(review) > 100 else review,
                    "sentiment": result["sentiment"],
                    "confidence": result["confidence"],
                }
            )

        usage = lm.get_usage()

        response = {
            "results": results,
            "total_reviews": len(reviews),
            "token_usage": {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            },
        }

        return json.dumps(response, indent=2)
    except Exception as e:
        error_response = {"error": f"{type(e).__name__}: {str(e)}"}
        return json.dumps(error_response, indent=2)


@function(sources=["Openaiconnector"])
def get_sentiment_summary(review: str) -> String:
    try:
        lm = CustomRESTLM(source_name="Openaiconnector", model="gpt-5-nano", temperature=1.0, max_tokens=20000)
        dspy.configure(lm=lm)

        lm.reset_usage()

        analyzer = MovieSentimentAnalyzer()
        result = analyzer.forward(review=review)

        sentiment = result["sentiment"]
        confidence = result["confidence"]

        usage = lm.get_usage()

        emoji_map = {"positive": "Happy", "negative": "Sad", "neutral": "Meh"}
        emoji = emoji_map.get(sentiment, "")

        summary = f"""
{emoji} Sentiment Analysis:
• Sentiment: {sentiment.upper()}
• Confidence: {confidence.upper()}
• Review: "{review[:100]}{"..." if len(review) > 100 else ""}"

Token Usage:
• Prompt Tokens: {usage["prompt_tokens"]}
• Completion Tokens: {usage["completion_tokens"]}
• Total Tokens: {usage["total_tokens"]}
        """.strip()

        return summary
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"
