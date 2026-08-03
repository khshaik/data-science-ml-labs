"""
Latency and throughput benchmarking script
Measures API performance for Section C requirements

Section C: Serving & Inference Pattern (25%) - Performance Measurement
"""

import requests
import time
import numpy as np
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LatencyBenchmark:
    """
    Benchmark API latency and throughput
    
    Requirements from Instructions.txt:
    - Report avg latency and p95 (even from a small run)
    - Simple script that sends multiple requests
    """
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.predict_endpoint = f"{api_url}/predict"
    
    def get_sample_request(self) -> dict:
        """
        Get sample customer data for prediction
        """
        return {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 65.50,
            "TotalCharges": 786.00
        }
    
    def single_request(self, request_id: int) -> dict:
        """
        Send single prediction request and measure latency
        """
        start_time = time.time()
        
        try:
            response = requests.post(
                self.predict_endpoint,
                json=self.get_sample_request(),
                timeout=5
            )
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                return {
                    'request_id': request_id,
                    'latency_ms': latency,
                    'success': True,
                    'status_code': response.status_code
                }
            else:
                return {
                    'request_id': request_id,
                    'latency_ms': latency,
                    'success': False,
                    'status_code': response.status_code
                }
        
        except Exception as e:
            return {
                'request_id': request_id,
                'latency_ms': -1,
                'success': False,
                'error': str(e)
            }
    
    def benchmark_sequential(self, num_requests: int = 200) -> dict:
        """
        Benchmark with sequential requests
        Measures average latency
        """
        logger.info(f"\n{'='*80}")
        logger.info("SEQUENTIAL LATENCY BENCHMARK")
        logger.info(f"{'='*80}")
        logger.info(f"Number of requests: {num_requests}")
        
        # Warmup request
        logger.info("\nWarmup request...")
        warmup_result = self.single_request(0)
        logger.info(f"Warmup latency: {warmup_result['latency_ms']:.2f} ms")
        
        # Benchmark requests
        logger.info(f"\nSending {num_requests} sequential requests...")
        results = []
        start_time = time.time()
        
        for i in range(1, num_requests + 1):
            result = self.single_request(i)
            results.append(result)
            
            if i % 50 == 0:
                logger.info(f"  Progress: {i}/{num_requests}")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        latencies = [r['latency_ms'] for r in results if r['success']]
        success_count = sum(1 for r in results if r['success'])
        
        metrics = {
            'num_requests': num_requests,
            'success_count': success_count,
            'success_rate': success_count / num_requests,
            'total_time_sec': total_time,
            'avg_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'std_latency_ms': np.std(latencies)
        }
        
        # Print results
        logger.info(f"\n{'='*80}")
        logger.info("SEQUENTIAL BENCHMARK RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Requests sent:      {metrics['num_requests']}")
        logger.info(f"Successful:         {metrics['success_count']} ({metrics['success_rate']*100:.1f}%)")
        logger.info(f"Total time:         {metrics['total_time_sec']:.2f} seconds")
        logger.info(f"\nLatency Statistics:")
        logger.info(f"  Average:          {metrics['avg_latency_ms']:.2f} ms")
        logger.info(f"  Median (p50):     {metrics['p50_latency_ms']:.2f} ms")
        logger.info(f"  p95:              {metrics['p95_latency_ms']:.2f} ms")
        logger.info(f"  p99:              {metrics['p99_latency_ms']:.2f} ms")
        logger.info(f"  Min:              {metrics['min_latency_ms']:.2f} ms")
        logger.info(f"  Max:              {metrics['max_latency_ms']:.2f} ms")
        logger.info(f"  Std Dev:          {metrics['std_latency_ms']:.2f} ms")
        logger.info(f"{'='*80}")
        
        return metrics
    
    def benchmark_concurrent(self, num_requests: int = 200, concurrency: int = 10) -> dict:
        """
        Benchmark with concurrent requests
        Measures throughput
        """
        logger.info(f"\n{'='*80}")
        logger.info("CONCURRENT THROUGHPUT BENCHMARK")
        logger.info(f"{'='*80}")
        logger.info(f"Number of requests: {num_requests}")
        logger.info(f"Concurrency:        {concurrency}")
        
        # Benchmark requests
        logger.info(f"\nSending {num_requests} concurrent requests...")
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(self.single_request, i) for i in range(num_requests)]
            
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                
                if i % 50 == 0:
                    logger.info(f"  Progress: {i}/{num_requests}")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        latencies = [r['latency_ms'] for r in results if r['success']]
        success_count = sum(1 for r in results if r['success'])
        
        metrics = {
            'num_requests': num_requests,
            'concurrency': concurrency,
            'success_count': success_count,
            'success_rate': success_count / num_requests,
            'total_time_sec': total_time,
            'throughput_rps': num_requests / total_time,
            'avg_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99)
        }
        
        # Print results
        logger.info(f"\n{'='*80}")
        logger.info("CONCURRENT BENCHMARK RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Requests sent:      {metrics['num_requests']}")
        logger.info(f"Concurrency:        {metrics['concurrency']}")
        logger.info(f"Successful:         {metrics['success_count']} ({metrics['success_rate']*100:.1f}%)")
        logger.info(f"Total time:         {metrics['total_time_sec']:.2f} seconds")
        logger.info(f"Throughput:         {metrics['throughput_rps']:.2f} requests/sec")
        logger.info(f"\nLatency Statistics:")
        logger.info(f"  Average:          {metrics['avg_latency_ms']:.2f} ms")
        logger.info(f"  p50:              {metrics['p50_latency_ms']:.2f} ms")
        logger.info(f"  p95:              {metrics['p95_latency_ms']:.2f} ms")
        logger.info(f"  p99:              {metrics['p99_latency_ms']:.2f} ms")
        logger.info(f"{'='*80}")
        
        return metrics
    
    def save_results(self, sequential_metrics: dict, concurrent_metrics: dict, output_file: str):
        """
        Save benchmark results to file
        """
        results = {
            'sequential': sequential_metrics,
            'concurrent': concurrent_metrics,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✅ Results saved to {output_file}")


def main():
    """
    Main benchmarking script
    """
    parser = argparse.ArgumentParser(description='Benchmark API latency and throughput')
    parser.add_argument('--url', default='http://localhost:8000', help='API URL')
    parser.add_argument('--requests', type=int, default=200, help='Number of requests')
    parser.add_argument('--concurrency', type=int, default=10, help='Concurrent requests')
    parser.add_argument('--output', default='artifacts/benchmark_results.json', help='Output file')
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = LatencyBenchmark(api_url=args.url)
    
    # Check API health
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code != 200:
            logger.error(f"API health check failed: {response.status_code}")
            return
        logger.info("✅ API is healthy")
    except Exception as e:
        logger.error(f"Cannot connect to API: {e}")
        logger.error("Please start the API server first: uvicorn src.serving.api:app")
        return
    
    # Run benchmarks
    sequential_metrics = benchmark.benchmark_sequential(num_requests=args.requests)
    concurrent_metrics = benchmark.benchmark_concurrent(num_requests=args.requests, concurrency=args.concurrency)
    
    # Save results
    benchmark.save_results(sequential_metrics, concurrent_metrics, args.output)
    
    logger.info("\n✅ Benchmarking completed successfully!")


if __name__ == "__main__":
    main()
