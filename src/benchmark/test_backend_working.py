import time
from loguru import logger
from openai import OpenAI

QUESTION = "What is Deep Learning?"

def check_answer(answer: str | None):
    assert answer is not None, "Generated text is empty"
    assert len(answer) > 0, "Generated text is empty"

def try_chat_request(model_id: str):
    
    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="-"
    )

    start_time = time.time()
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": QUESTION}
        ]
    )
    end_time = time.time()
    duration = end_time - start_time

    answer = completion.choices[0].message.content
    
    logger.info(f"Answer received in {duration:.2f} seconds: {answer}")
    
    check_answer(answer)

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
        logger.error(f"Chat request failed. Error: {str(e)}")
        
    return False

def test_backend_working(model_id: str) -> bool:
    """
    Try a single request 3 times with exponential backoff.
    """
    
    logger.info(f"Testing if the backend is working with model {model_id}")
    
    retries = 3
    base_delay = 5  # Start with 1 second delay
    
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