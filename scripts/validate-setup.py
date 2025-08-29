#!/usr/bin/env python3
"""
Validation script to test the project setup
"""
import sys
import importlib.util
import os

def test_shared_imports():
    """Test that shared utilities can be imported (syntax check only)"""
    print("Testing shared utilities syntax...")
    
    try:
        # Test that files exist and have valid Python syntax
        shared_files = ["config.py", "database.py", "logging.py"]
        
        for file in shared_files:
            file_path = f"shared/{file}"
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Basic syntax check by compiling
                compile(content, file_path, 'exec')
                print(f"✓ {file} has valid Python syntax")
            else:
                print(f"✗ {file} missing")
                return False
        
        print("Note: Full import testing requires dependencies (will work in Docker containers)")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in {e.filename}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking shared utilities: {e}")
        return False

def test_service_structure():
    """Test that all services have required files"""
    print("\nTesting service structure...")
    
    services = [
        "story-service",
        "tts-service", 
        "image-service",
        "music-service",
        "video-service",
        "moderation-service"
    ]
    
    required_files = ["main.py", "requirements.txt", "Dockerfile"]
    all_good = True
    
    for service in services:
        service_path = f"services/{service}"
        if not os.path.exists(service_path):
            print(f"✗ Service directory missing: {service_path}")
            all_good = False
            continue
            
        for file in required_files:
            file_path = f"{service_path}/{file}"
            if os.path.exists(file_path):
                print(f"✓ {service}/{file} exists")
            else:
                print(f"✗ {service}/{file} missing")
                all_good = False
    
    return all_good

def test_docker_compose():
    """Test that docker-compose.yml exists and is valid"""
    print("\nTesting Docker Compose configuration...")
    
    if os.path.exists("docker-compose.yml"):
        print("✓ docker-compose.yml exists")
        
        # Basic validation - check if it contains expected services
        with open("docker-compose.yml", "r") as f:
            content = f.read()
            
        expected_services = [
            "postgres", "redis", "story-service", "tts-service",
            "image-service", "music-service", "video-service", "moderation-service"
        ]
        
        all_services_found = True
        for service in expected_services:
            if service in content:
                print(f"✓ {service} found in docker-compose.yml")
            else:
                print(f"✗ {service} missing from docker-compose.yml")
                all_services_found = False
        
        return all_services_found
    else:
        print("✗ docker-compose.yml missing")
        return False

def main():
    """Run all validation tests"""
    print("AI Story-to-Video Pipeline Setup Validation")
    print("=" * 50)
    
    tests = [
        test_shared_imports,
        test_service_structure,
        test_docker_compose
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✓ All validation tests passed!")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and configure your API keys")
        print("2. Run 'docker-compose up -d' to start all services")
        print("3. Run 'make health' to check service health")
        return 0
    else:
        print("✗ Some validation tests failed!")
        print("Please fix the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())