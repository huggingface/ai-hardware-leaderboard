import requests
import time
from loguru import logger
from huggingface_hub import InferenceClient
from huggingface_hub.utils._auth import get_token
from typing import Dict, Union
import json

QUESTION = "What is Deep Learning?"

# def check_answer_is_sensible(answer: str) -> bool:
#     client = InferenceClient(
#         provider="hf-inference",
#         api_key=get_token()
#     )

#     # Define the function schema that we want the model to use
#     functions = [
#         {
#             "name": "evaluate_answer",
#             "description": "Evaluate if an answer is sensible for a given question",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "is_sensible": {
#                         "type": "boolean",
#                         "description": "Whether the answer is sensible and relevant to the question"
#                     },
#                     "reason": {
#                         "type": "string",
#                         "description": "Explanation for why the answer is or isn't sensible"
#                     }
#                 },
#                 "required": ["is_sensible", "reason"]
#             }
#         }
#     ]

#     messages = [
#         {
#             "role": "system",
#             "content": "You are a helpful assistant that evaluates answers. Always respond with valid JSON."
#         },
#         {
#             "role": "user",
#             "content": f"Evaluate if this answer is sensible for the question.\nQuestion: {QUESTION}\nAnswer: {answer}"
#         }
#     ]

#     completion = client.chat.completions.create(
#         model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
#         messages=messages,
#         functions=functions,
#         function_call={"name": "evaluate_answer"},
#         response_format={ "type": "json_object" }  # Force JSON mode
#     )
    
#     # Parse the function call response
#     function_args = completion.choices[0].message.function_call.arguments

#     try:
#         result = json.loads(function_args)
            
#     except Exception as e:
#         logger.error(f"Error evaluating answer: {str(e)}")
#         return {
#             "is_sensible": False,
#             "reason": f"Error during evaluation: {str(e)}"
#         }

#     assert 'is_sensible' in result, "is_sensible is not in the result"
#     assert 'reason' in result, "reason is not in the result"
    
#     if not result['is_sensible']:
#         logger.error("The response by the endpoint is non sensical")
    
#     return result['is_sensible']

from openai import OpenAI

client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="-"
    )

def check_answer(answer: str | None):
    assert answer is not None, "Generated text is empty"
    assert len(answer) > 0, "Generated text is empty"

def try_chat_request(model_id: str):
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": QUESTION}
        ]
    )

    answer = completion.choices[0].message.content
    
    check_answer(answer)

def try_completion_request(model_id: str):
    completion = client.completions.create(
        model=model_id,
        prompt=QUESTION,
        max_tokens=20
    )
    
    answer = completion.choices[0].text
    check_answer(answer)

    return answer



def try_single_request(model_id: str) -> bool:
    """
    Try a single request to an openai api to check if it is working.
    First attempts a chat request, and if that fails, falls back to a completion request.
    Returns True if either request succeeds, False if both fail.
    """
    
    # First try chat request
    try: 
        try_chat_request(model_id)
        logger.info("Chat request succeeded")
        return True
    except Exception as e:
        logger.warning(f"Chat request failed, falling back to completion request. Error: {str(e)}")
    
    # If chat fails, try completion request
    try:
        try_completion_request(model_id)
        logger.info("Completion request succeeded")
        return True
    except Exception as e:
        logger.error(f"Both chat and completion requests failed. Completion error: {str(e)}")
    
    return False

def test_backend_working(model_id: str) -> bool:
    """
    Try a single request 3 times with exponential backoff.
    """
    retries = 3
    base_delay = 1  # Start with 1 second delay
    
    for attempt in range(retries):
        try:
            try_single_request(model_id)
            return True  # Success, exit function
        except AssertionError:
            if attempt == retries - 1:  # Last attempt
                raise  # Re-raise the last error
            delay = base_delay * (2 ** attempt)  # Exponential backoff
            time.sleep(delay)
            continue
        
    return False