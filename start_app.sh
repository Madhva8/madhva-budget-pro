#\!/bin/bash

# Terminal colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Display startup banner
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}   Madhva Budget Pro - Linux Start   ${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Change to the src directory
echo -e "Working directory: ${GREEN}$DIR${NC}"
echo -e "Changing to: ${GREEN}$DIR/src${NC}"
cd "$DIR/src"

# Find Python
PYTHON=$(which python3)
if [ -z "$PYTHON" ]; then
    echo -e "${RED}Error: Python 3 not found. Please install Python 3.${NC}"
    echo "Press Enter to exit..."
    read
    exit 1
fi

echo -e "Using Python: ${GREEN}$PYTHON${NC}"
echo -e "Python version: ${GREEN}$($PYTHON --version)${NC}"
echo ""

# Check database
if [ \! -f "$DIR/financial_planner.db" ] && [ -f "$DIR/financial_planner.db.sample" ]; then
    echo -e "${BLUE}Database not found. Setting up from sample...${NC}"
    cp "$DIR/financial_planner.db.sample" "$DIR/financial_planner.db"
    
    # Add sample data if the script exists
    if [ -f "$DIR/fix_pie_chart.py" ]; then
        echo -e "${BLUE}Adding sample expense data...${NC}"
        cd "$DIR"
        "$PYTHON" fix_pie_chart.py
        cd "$DIR/src"
    fi
fi

echo -e "${BLUE}Starting Madhva Budget Pro...${NC}"
echo ""

# Run the application
"$PYTHON" main_pyside6.py

# Check if application crashed
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo -e "${RED}The application encountered an error (Exit code: $EXIT_CODE).${NC}"
    echo ""
    echo -e "${BLUE}Diagnostic information:${NC}"
    echo -e "${BLUE}----------------------${NC}"
    
    # Check dependencies
    echo "Checking required packages:"
    for pkg in PySide6 matplotlib pandas pdfplumber numpy; do
        if $PYTHON -c "import $pkg" 2>/dev/null; then
            echo -e "$pkg: ${GREEN}Installed${NC}"
        else
            echo -e "$pkg: ${RED}Not found${NC}"
        fi
    done
    
    echo ""
    echo "To install all dependencies, run: pip3 install -r requirements.txt"
    echo ""
    echo "Press Enter to exit..."
    read
fi

exit $EXIT_CODE
