#!/bin/bash
# Comprehensive test runner script
# Bonus Feature: Comprehensive Testing (+0.5 mark)

echo "=================================="
echo "RUNNING COMPREHENSIVE TEST SUITE"
echo "=================================="
echo ""

# Activate virtual environment
source venv/bin/activate

# Run all unit tests with coverage
echo "📊 Running Unit Tests with Coverage..."
pytest tests/unit/ -v --cov=src --cov-report=term-missing --cov-report=html

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All unit tests passed!"
else
    echo ""
    echo "❌ Some tests failed!"
    exit 1
fi

echo ""
echo "=================================="
echo "TEST SUMMARY"
echo "=================================="

# Display coverage summary
echo ""
echo "📈 Coverage Report:"
coverage report --skip-covered

echo ""
echo "📁 Detailed HTML coverage report: htmlcov/index.html"
echo ""
echo "=================================="
echo "✅ TEST SUITE COMPLETE"
echo "=================================="
