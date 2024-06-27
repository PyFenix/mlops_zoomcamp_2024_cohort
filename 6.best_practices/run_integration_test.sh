#!/bin/bash

# Step 1: Start Localstack using Docker Compose
docker compose up -d

# Wait for Localstack to be ready
echo "Waiting for Localstack to be ready..."
sleep 15

# Step 2: Create the S3 bucket
aws s3 mb s3://nyc-duration --endpoint-url=http://localhost:4566

# Step 3: Run the integration test script
python integration_test.py

# Step 4: Shut down Localstack
docker compose down

# Display the results
echo "Integration test completed. Check check check it out!"
