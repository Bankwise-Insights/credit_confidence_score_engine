# performance_test.py
import requests
import time
import random
import json

API_URL = "http://127.0.0.1:8000/predict_credit_risk"
HEADERS = {"Content-Type": "application/json"}

# --- Generate diverse test data ---
def generate_random_credit_data():
    """Generates a random but plausible credit application payload."""
    return {
        "Age": random.randint(20, 65),
        "Income": round(random.uniform(30000.0, 150000.0), 2),
        "MonthsEmployed": random.randint(1, 480), # Up to 40 years
        "DTIRatio": round(random.uniform(0.05, 0.60), 2),
        "Education": random.choice(['Primary', 'Secondary', 'Undergraduate', 'Postgraduate']),
        "EmploymentType": random.choice(['Unemployed', 'Salaried', 'Self-Employed', 'Contract-Part-time']),
        "MaritalStatus": random.choice(['Single', 'Married', 'Divorced/Widowed']),
        "HasMortgage": random.choice([0, 1]),
        "HasDependents": random.randint(0, 5),
        "LoanPurpose": random.choice(['Debt Consolidation', 'Home Improvement', 'Business', 'Education', 'Other-Miscellaneous']),
        "HasCoSigner": random.choice([0, 1])
    }

# --- Performance Test Parameters ---
NUM_REQUESTS = 500  # Number of requests to send
CONCURRENT_REQUESTS = 10 # Number of concurrent requests (requires aiohttp or threading.ThreadPoolExecutor for true concurrency)
                          # For this simple script, we'll send them sequentially but record individual times.
                          # For true concurrency, you'd use libraries like `concurrent.futures` or `httpx`/`aiohttp`.

response_times_ms = []
successful_requests = 0
failed_requests = 0

print(f"Starting performance test: Sending {NUM_REQUESTS} requests to {API_URL}")
print("-" * 50)

for i in range(NUM_REQUESTS):
    payload = generate_random_credit_data()

    start_time = time.time()
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        end_time = time.time()

        response_time_ms = (end_time - start_time) * 1000
        response_times_ms.append(response_time_ms)
        successful_requests += 1

        # Optional: Print some details for the first few requests
        if i < 5 or i % 100 == 0:
            print(f"Request {i+1} successful. Status: {response.status_code}, Time: {response_time_ms:.2f}ms")
            # print(f"  Prediction: {response.json().get('credit_risk_prediction')}, Prob: {response.json().get('probability_high_risk'):.4f}")
    except requests.exceptions.RequestException as e:
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000 # Still record time for failures if desired
        response_times_ms.append(response_time_ms) # Include failed request times in avg
        failed_requests += 1
        print(f"Request {i+1} FAILED. Error: {e}, Time: {response_time_ms:.2f}ms")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response content: {e.response.text}")

    # Optional: Add a small delay if you don't want to bombard the server too fast
    # time.sleep(0.01)

print("-" * 50)
print("\n--- Performance Test Results ---")
print(f"Total requests attempted: {NUM_REQUESTS}")
print(f"Successful requests: {successful_requests}")
print(f"Failed requests: {failed_requests}")

if successful_requests > 0:
    avg_response_time = sum(response_times_ms) / len(response_times_ms) # Calculate over all requests, including failed if logged
    min_response_time = min(response_times_ms)
    max_response_time = max(response_times_ms)

    # Sort times to calculate percentiles
    sorted_times = sorted(response_times_ms)
    p50 = sorted_times[int(len(sorted_times) * 0.50)]
    p90 = sorted_times[int(len(sorted_times) * 0.90)]
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    p99 = sorted_times[int(len(sorted_times) * 0.99)]

    print(f"Average Response Time: {avg_response_time:.2f} ms")
    print(f"Minimum Response Time: {min_response_time:.2f} ms")
    print(f"Maximum Response Time: {max_response_time:.2f} ms")
    print(f"50th Percentile (P50) Response Time: {p50:.2f} ms")
    print(f"90th Percentile (P90) Response Time: {p90:.2f} ms")
    print(f"95th Percentile (P95) Response Time: {p95:.2f} ms")
    print(f"99th Percentile (P99) Response Time: {p99:.2f} ms") 
else:
    print("No successful requests were made.")

print("Note: This script sends requests sequentially. For true concurrent load testing, consider tools like Apache JMeter, Locust, or Python's `httpx` with `asyncio`.")