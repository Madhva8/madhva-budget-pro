#!/bin/bash

# This is the most direct approach - just run the app directly
# Do not skip login to maintain security
cd "$(dirname "$0")"
python3 src/main_pyside6.py