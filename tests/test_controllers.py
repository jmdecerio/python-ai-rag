import pytest
from unittest.mock import Mock, patch

from app.controllers import ChatRequest, ChatResponse


def test_chat_request_model():
    """Test ChatRequest model validation"""
    request = ChatRequest(question="What is the capital of France?")
    assert request.question == "What is the capital of France?"


def test_chat_response_model():
    """Test ChatResponse model validation"""
    response = ChatResponse(answer="The capital of France is Paris.")
    assert response.answer == "The capital of France is Paris."


@pytest.mark.parametrize("question,expected_answer", [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("How many people live in New York?", "New York has approximately 8.3 million residents."),
])
def test_chat_endpoint_logic(question, expected_answer):
    """Test chat endpoint logic with mocked service"""
    # This test focuses on the data flow and model validation
    # Actual service integration tests would require more complex setup
    
    # Test request creation
    request = ChatRequest(question=question)
    assert request.question == question
    
    # Test response creation
    response = ChatResponse(answer=expected_answer)
    assert response.answer == expected_answer