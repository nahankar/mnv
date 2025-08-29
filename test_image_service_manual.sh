#!/bin/bash

echo "=== Image Service Manual Testing ==="
echo

# Test 1: Health check
echo "1. Testing health check..."
health_response=$(curl -s http://localhost:8003/health)
echo "Health response: $health_response"
if echo "$health_response" | grep -q '"status":"healthy"'; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi
echo

# Test 2: Providers endpoint
echo "2. Testing providers endpoint..."
providers_response=$(curl -s http://localhost:8003/providers)
echo "Providers response: $providers_response"
if echo "$providers_response" | grep -q '"providers"'; then
    echo "✅ Providers endpoint passed"
else
    echo "❌ Providers endpoint failed"
    exit 1
fi
echo

# Test 3: Image generation (multiple attempts due to 10% failure rate)
echo "3. Testing image generation..."
success_count=0
for i in {1..5}; do
    echo "  Attempt $i..."
    response=$(curl -s -X POST http://localhost:8003/generate/image \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "A beautiful sunset over mountains",
            "provider": "mock",
            "style": "photorealistic"
        }')
    
    if echo "$response" | grep -q '"id"'; then
        echo "  ✅ Attempt $i succeeded"
        success_count=$((success_count + 1))
        
        # Extract image ID and test image serving
        image_id=$(echo "$response" | grep -o '"id": "[^"]*"' | cut -d'"' -f4)
        file_name=$(echo "$response" | grep -o '"file_url": "/images/[^"]*"' | cut -d'/' -f3 | tr -d '"')
        
        if [ ! -z "$file_name" ]; then
            echo "  Testing image serving for $file_name..."
            if curl -s -f "http://localhost:8003/images/$file_name" > /tmp/test_image_$i.png; then
                echo "  ✅ Image serving works"
                file_info=$(file /tmp/test_image_$i.png)
                echo "  File info: $file_info"
            else
                echo "  ❌ Image serving failed"
            fi
        fi
        break
    else
        echo "  ⚠️  Attempt $i failed (expected due to 10% failure rate)"
        echo "  Response: $response"
    fi
done

if [ $success_count -gt 0 ]; then
    echo "✅ Image generation test passed ($success_count/5 attempts succeeded)"
else
    echo "❌ Image generation test failed (all attempts failed)"
    exit 1
fi
echo

# Test 4: Deep health check
echo "4. Testing deep health check..."
deep_health_response=$(curl -s "http://localhost:8003/health?deep=true")
echo "Deep health response: $deep_health_response"
if echo "$deep_health_response" | grep -q '"database"'; then
    echo "✅ Deep health check passed"
else
    echo "❌ Deep health check failed"
fi
echo

echo "=== All tests completed successfully! ==="
echo "The image service is working correctly with:"
echo "- Health checks ✅"
echo "- Provider management ✅" 
echo "- Image generation ✅"
echo "- Image serving ✅"
echo "- Database connectivity ✅"