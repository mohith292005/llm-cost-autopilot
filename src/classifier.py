from __future__ import annotations

import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_PATH = Path("classifier.joblib")

TRAINING_DATA: list[tuple[str, int]] = [
    # Tier 1: Simple
    ("Convert this date to ISO format: July 4, 2024", 1),
    ("What is the capital of France?", 1),
    ("Translate 'hello world' to Spanish", 1),
    ("Extract all email addresses from this text: contact us at foo@bar.com", 1),
    ("Is this sentence grammatically correct? 'She go to school'", 1),
    ("List the days of the week", 1),
    ("What does API stand for?", 1),
    ("Format this JSON prettily: {'a':1,'b':2}", 1),
    ("Convert 100 Fahrenheit to Celsius", 1),
    ("What is 15% of 240?", 1),
    ("Correct the spelling: 'recieve the messege tommorow'", 1),
    ("What color is the sky?", 1),
    ("Remove duplicate words: the the cat sat sat on", 1),
    ("Is 'racecar' a palindrome?", 1),
    ("Sort this list alphabetically: banana, apple, cherry, date", 1),
    ("What HTTP status code means 'Not Found'?", 1),
    ("Extract the domain from this URL: https://www.example.com/page", 1),
    ("How many days are in February 2024?", 1),
    ("Capitalize the first letter of each word: hello world foo bar", 1),
    ("What is the plural of 'mouse'?", 1),

    # Tier 2: Moderate
    ("Summarize this 500-word article about climate change in 3 bullet points", 2),
    ("Write a professional email declining a job offer politely", 2),
    ("Compare the pros and cons of React vs Vue for a small team", 2),
    ("Explain what a REST API is to a junior developer", 2),
    ("Analyze the sentiment and key themes in this customer review paragraph", 2),
    ("Create a structured project plan for building a landing page", 2),
    ("Write a LinkedIn post announcing a new product feature launch", 2),
    ("Rewrite this paragraph to be more concise and professional", 2),
    ("Generate 5 creative names for a SaaS expense tracking startup", 2),
    ("Explain the difference between SQL joins with a simple example", 2),
    ("Summarize the key differences between Docker and virtual machines", 2),
    ("Write a test case description for a user login feature", 2),
    ("Create a weekly meal plan for a vegetarian athlete", 2),
    ("Explain Git branching strategy for a small development team", 2),
    ("Write documentation for a simple Python function that adds two numbers", 2),
    ("Analyze the strengths and weaknesses of this business model description", 2),
    ("Generate interview questions for a senior Python developer role", 2),
    ("Summarize the main arguments in this research abstract", 2),
    ("Create a checklist for deploying a web application to production", 2),
    ("Write a product requirements document for a notification system", 2),

    # Tier 3: Complex
    ("Design a distributed rate limiter for 1 million requests per second using Redis", 3),
    ("Write a Python async web scraper that handles pagination and rate limiting", 3),
    ("Implement a binary search tree with insert, delete, and in-order traversal in Python", 3),
    ("Debug this race condition in my async Python code and propose a fix with explanation", 3),
    ("Design the database schema for a multi-tenant SaaS application with row-level security", 3),
    ("Write a complete FastAPI endpoint with JWT auth, input validation, and error handling", 3),
    ("Explain how to implement CQRS and event sourcing for an e-commerce order system", 3),
    ("Design a CI/CD pipeline for a microservices app with blue-green deployments", 3),
    ("Implement a LRU cache from scratch in Python with O(1) get and put operations", 3),
    ("Write a React component with useReducer managing complex multi-step form state", 3),
    ("Analyze the time and space complexity of this algorithm and suggest optimizations", 3),
    ("Design an API gateway with authentication, rate limiting, and request transformation", 3),
    ("Write a Kubernetes deployment YAML for a stateful application with persistent volumes", 3),
    ("Implement a pub/sub event system using asyncio queues with backpressure handling", 3),
    ("Create a full data pipeline: ingest CSV, transform, validate, and load to PostgreSQL", 3),
    ("Debug and refactor this 100-line Python class with multiple design pattern violations", 3),
    ("Design a caching strategy for a social media feed with 10 million daily active users", 3),
    ("Write comprehensive unit tests for a payment processing module using pytest and mocks", 3),
    ("Implement OAuth2 PKCE flow from scratch explaining each step and security rationale", 3),
    ("Architect a real-time collaborative document editing system like Google Docs", 3),
]


class ComplexityClassifier:
    def __init__(self) -> None:
        self.pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ])
        self._is_trained = False

    def train(self, data: list[tuple[str, int]] | None = None) -> dict:
        if data is None:
            data = TRAINING_DATA

        texts  = [item[0] for item in data]
        labels = [item[1] for item in data]

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        self.pipeline.fit(X_train, y_train)
        self._is_trained = True

        y_pred = self.pipeline.predict(X_test)
        accuracy = float(np.mean(np.array(y_pred) == np.array(y_test)))

        print(f"\n{'='*50}")
        print(f"  Classifier trained! Accuracy: {accuracy:.1%}")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred, target_names=["Simple", "Moderate", "Complex"]))

        return {"accuracy": accuracy, "test_size": len(y_test)}

    def predict(self, text: str) -> tuple[int, float]:
        if not self._is_trained:
            raise RuntimeError("Classifier not trained. Call .train() or .load() first.")

        proba = self.pipeline.predict_proba([text])[0]
        tier = int(self.pipeline.predict([text])[0])
        confidence = float(max(proba))

        return tier, confidence

    def save(self, path: Path = MODEL_PATH) -> None:
        if not self._is_trained:
            raise RuntimeError("Nothing to save — model not trained yet.")
        joblib.dump(self.pipeline, path)
        print(f"  ✓ Model saved to {path} ({path.stat().st_size / 1024:.1f} KB)")

    def load(self, path: Path = MODEL_PATH) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Run: python src/classifier.py"
            )
        self.pipeline = joblib.load(path)
        self._is_trained = True
        print(f"  ✓ Classifier loaded from {path}")


_classifier_instance: ComplexityClassifier | None = None


def get_classifier() -> ComplexityClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ComplexityClassifier()
        try:
            _classifier_instance.load()
        except FileNotFoundError:
            print("  No saved classifier found. Training from scratch...")
            _classifier_instance.train()
            _classifier_instance.save()
    return _classifier_instance


def train_and_save() -> None:
    c = ComplexityClassifier()
    c.train()
    c.save()


if __name__ == "__main__":
    print("\nTraining classifier...")
    train_and_save()

    print("\nTesting predictions:")
    clf = get_classifier()
    test_prompts = [
        "What is 2 + 2?",
        "Summarize this article about machine learning trends",
        "Design a distributed database with eventual consistency guarantees",
    ]
    for prompt in test_prompts:
        tier, conf = clf.predict(prompt)
        labels = {1: "Simple 🟢", 2: "Moderate 🟡", 3: "Complex 🔴"}
        print(f"  [{labels[tier]} | {conf:.0%} conf] {prompt[:60]}")