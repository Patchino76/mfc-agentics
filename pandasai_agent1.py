from pandasai import PandasAI
from pandasai.llm import OpenAI
from pandasai.helpers.security import SecurityHandler
from typing import List, Optional
from pydantic import BaseModel
import pandas as pd

# Define custom security settings
class CustomSecurityHandler(SecurityHandler):
    def is_import_allowed(self, module_name: str) -> bool:
        # Define allowed imports
        ALLOWED_IMPORTS = {
            'pandas', 'numpy', 'io', 'datetime', 
            'math', 'statistics', 'typing',
            'collections', 'itertools'
        }
        return module_name in ALLOWED_IMPORTS

    def is_attribute_allowed(self, attribute_name: str) -> bool:
        # Define any restricted attributes
        RESTRICTED_ATTRIBUTES = {'system', 'subprocess', 'os'}
        return attribute_name not in RESTRICTED_ATTRIBUTES

# Your Pydantic models
class DepartmentStats(BaseModel):
    department: str
    avg_salary: float
    employee_count: int
    min_salary: float
    max_salary: float

class DepartmentAnalysis(BaseModel):
    departments: List[DepartmentStats]
    total_employees: int
    company_avg_salary: float

def setup_pandasai():
    # Initialize the LLM
    llm = OpenAI(api_token="your-api-key")
    
    # Configure PandasAI with custom security
    pandas_ai = PandasAI(
        llm=llm,
        security_handler=CustomSecurityHandler(),
        enforce_privacy=True,
        verbose=True  # Helpful for debugging
    )
    return pandas_ai

def analyze_with_safety(df: pd.DataFrame) -> DepartmentAnalysis:
    pandas_ai = setup_pandasai()
    
    try:
        # Run analysis with error handling
        result = pandas_ai.run(
            df,
            """
            Analyze the department statistics and return a dictionary with:
            - List of department stats (avg_salary, employee_count, min/max salary)
            - Total employee count
            - Company-wide average salary
            Format the response as a dictionary matching the DepartmentAnalysis structure.
            """,
            return_type=dict  # Specify return type explicitly
        )
        
        return DepartmentAnalysis.parse_obj(result)
    
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        # You might want to implement retry logic here
        raise

# Example usage
def main():
    # Sample data
    df = pd.DataFrame({
        'department': ['Engineering', 'Sales', 'Engineering', 'Marketing'],
        'salary': [90000, 75000, 85000, 70000],
        'employee_id': [1, 2, 3, 4]
    })

    try:
        analysis = analyze_with_safety(df)
        
        # Print results
        print(f"\nCompany Overview:")
        print(f"Total Employees: {analysis.total_employees}")
        print(f"Company Average Salary: ${analysis.company_avg_salary:,.2f}")
        
        print("\nDepartment Breakdown:")
        for dept in analysis.departments:
            print(f"\n{dept.department}:")
            print(f"  Average Salary: ${dept.avg_salary:,.2f}")
            print(f"  Employees: {dept.employee_count}")
            print(f"  Salary Range: ${dept.min_salary:,.2f} - ${dept.max_salary:,.2f}")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()