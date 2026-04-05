import requests
import json

# Your live Hugging Face URL
BASE_URL = "https://ishikamahadar-resume-env.hf.space"

def test_environment():
    print(f"--- Testing Environment at: {BASE_URL} ---")
    
    # 1. Reset
    print("\n[1/2] Resetting environment...")
    reset_resp = requests.post(f"{BASE_URL}/reset")
    print(f"Full Reset Response: {json.dumps(reset_resp.json(), indent=2)}")
    
    if reset_resp.status_code == 200:
        obs_data = reset_resp.json()
        # Handle potential nesting in OpenEnv
        obs = obs_data.get('observation', obs_data)
        print("✅ Reset Successful!")
        print(f"Task Type: {obs.get('task_type')}")
    else:
        print(f"❌ Reset Failed: {reset_resp.status_code}")
        return

    # 2. Step
    print("\n[2/2] Submitting action...")
    action_data = {
        "action": {
            "decision": "accept",
            "fraud_flag": False,
            "confidence": 0.95
        }
    }
    step_resp = requests.post(f"{BASE_URL}/step", json=action_data)
    print(f"Server Status Code: {step_resp.status_code}")
    
    try:
        print(f"Full Step Response (JSON): {json.dumps(step_resp.json(), indent=2)}")
        if step_resp.status_code == 200:
            print("✅ Step Successful!")
    except Exception as e:
        print(f"❌ Could not parse JSON response. Raw Response Text:")
        print("-" * 20)
        print(step_resp.text)
        print("-" * 20)

if __name__ == "__main__":
    test_environment()
