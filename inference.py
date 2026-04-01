import os
import json
from openai import OpenAI
from client import ResumeEnv
from models import ResumeAction

def main():
    # 1. Read configuration from environment variables
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    model_name = os.getenv("MODEL_NAME", "gpt-4o")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return

    # 2. Initialize env and OpenAI client
    client = OpenAI(api_key=openai_api_key)
    env = ResumeEnv(base_url=api_base_url)

    # 3. reset()
    observation = env.reset()
    task_type = observation.task_type
    
    total_reward = 0.0
    done = False
    
    # 4. Inference Loop (One-step episode)
    while not done:
        # Prompt the LLM
        prompt = f"""
        You are a recruitment specialist. Evaluate the following resume against the job description.
        
        Job Description:
        {observation.job_description}
        
        Resume Text:
        {observation.resume_text}
        
        Provide your decision in JSON format:
        - decision: "accept" or "reject"
        - fraud_flag: true or false
        - confidence: a value between 0 and 1
        """

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        content = json.loads(response.choices[0].message.content)
        
        # 5. Create action
        action = ResumeAction(
            decision=content.get("decision", "reject"),
            fraud_flag=content.get("fraud_flag", False),
            confidence=content.get("confidence", 0.0)
        )

        # 6. Call env.step()
        observation, reward, done, info = env.step(action)
        
        # 7. Accumulate reward
        total_reward += reward

    # 8. Print results
    print(f"--- Episode Results ---")
    print(f"Task Type: {task_type}")
    print(f"Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
