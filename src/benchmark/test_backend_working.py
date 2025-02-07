import time
from loguru import logger
from openai import OpenAI

QUESTION = "What is Deep Learning?"



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
        if try_single_request(model_id):
            return True
        else:
            if attempt == retries - 1:  # Break if last attempt fails
                break
            
            # Exponential backoff
            delay = base_delay * (2 ** attempt)  
            time.sleep(delay)
            continue
        
    return False