# Matplotlib Custom Test Suite

This repository contains custom test cases developed for the Matplotlib project as part of a comprehensive testing strategy to improve code coverage and quality.

## Prerequisites

Before running the custom test suite, ensure you have a proper local development environment set up for Matplotlib.

## Setup Instructions

### 1. Environment Setup

Follow the official Matplotlib setup instructions for local environment testing:
- Visit: https://matplotlib.org/devdocs/devel/testing.html
- Complete all environment setup steps as outlined in the documentation
- Ensure all dependencies are properly installed

### 2. File Installation

Copy all custom test files to the Matplotlib library directory:

```bash
# Navigate to your matplotlib installation
cd /path/to/your/matplotlib/installation

# Copy all test files to the custom test directory
cp /path/to/custom/test/files/* ./lib/matplotlib/customtest/
```

**Important:** All test files must be placed in the `/lib/matplotlib/customtest/` directory for proper execution.

### 3. Directory Structure

After copying, your directory structure should look like:
```
matplotlib/
├── lib/
│   └── matplotlib/
│       └── customtest/
│           ├── test_*.py files
│           └── [other custom test files]
```

## Running the Tests

Once all files are properly copied to the specified path, execute the custom test suite:

```bash
# Run the custom test suite
pytest customtest
```

## Test Suite Information

This custom test suite includes:
- **1,009 custom test cases** targeting various Matplotlib components
- **98% code coverage** across 4,381 statements
- **Function-level testing** for utility functions
- **Class-level testing** for core components
- **Integration testing** for component interactions
- **Property-based testing** for edge cases

## Expected Results

The test suite should execute successfully and provide coverage reports showing improved test coverage across previously under-tested Matplotlib modules.
For the output evidence of these tests, please check ```Custom Test Cases Evidence.pdf```
