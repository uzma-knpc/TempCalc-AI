#gemini key set
import os
import streamlit as st
#from google.colab import userdata
import langchain_google_genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
import datetime
from langchain_core.tools import tool
import math
import statistics
from dotenv import load_dotenv


load_dotenv()

GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')



llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=GOOGLE_API_KEY)
@tool
def calculator(expressions):
    """
    A calculator function that evaluates mathematical expressions
    and formats the results in Google Doc style.

    Parameters:
    - expressions (list of str): A list of mathematical expressions to evaluate.

    Returns:
    - str: A formatted Google Doc-style report with the results.
    """

    # Google Doc-style header
    header = """
    ==========================================================================
                                Google Docs Style Calculator
    ==========================================================================
    """

    # Google Doc-style footer
    footer = f"""
    ==========================================================================
    Report Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ==========================================================================
    """

    # Initialize results
    results = []

    # Process each expression
    for idx, expression in enumerate(expressions, start=1):
        try:
            # Safely evaluate the mathematical expression
            result = eval(expression, {"__builtins__": {}}, {})
            results.append(f"{idx}. {expression} = {result}")
        except Exception as e:
            # If an error occurs, append the error message
            results.append(f"{idx}. {expression} = Error: {e}")

    # Combine header, results, and footer
    body = "\n".join(results)
    report = header + "\n" + body + "\n" + footer
    return report


# Example Usage


    # Generate the Google Doc-style report
    report =calculator(expressions)

    # Print the report
    print(report)

    # Save the report to a file
    #with open(calculator_report.txt", "w") as file:
     #  file.write(report)

    print("\nReport saved to 'calculator_report.txt'.")


#Advance maths tool

@tool
def advanced_calculator(expression: str, **kwargs):
    """
    Perform advanced mathematical operations including trigonometric calculations,
    statistics, and temperature conversions.

    Args:
        expression (str): A string containing a mathematical expression.
            Supports basic math operations (+, -, *, /), trigonometric functions (sin, cos, tan),
            statistical functions (mean, median, stdev), and temperature conversions (C_to_F, F_to_C).
        **kwargs: Additional arguments to handle dynamic variables in expressions.

    Returns:
        result (float): The result of the evaluated mathematical expression.

    Example:
        advanced_calculator("sin(math.pi / 2)") # Output: 1.0
        advanced_calculator("mean([1, 2, 3, 4])") # Output: 2.5
        advanced_calculator("C_to_F(25)") # Output: 77.0
    """

    # Custom functions for temperature conversion
    def C_to_F(celsius):
        """Convert Celsius to Fahrenheit."""
        return (celsius * 9/5) + 32

    def F_to_C(fahrenheit):
        """Convert Fahrenheit to Celsius."""
        return (fahrenheit - 32) * 5/9

    # Add custom functions and constants to the kwargs dictionary for eval
    kwargs.update({
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "mean": statistics.mean,
        "median": statistics.median,
        "stdev": statistics.stdev,
        "C_to_F": C_to_F,
        "F_to_C": F_to_C
    })

    try:
        # Evaluate the expression dynamically using eval
        result = eval(expression, {"__builtins__": None}, kwargs)
        return result
    except Exception as e:
        return f"Error in expression: {e}"

# Examples

#tools=[Multiplication,addition,module,Subtraction,addition]
tools=[calculator,advanced_calculator]

#agent=initialize_agent(tools,llm,agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)

#response=agent.invoke({"input":"mean and standard deviation of 1,5,6,8 "})
#print(f"\n{response}\n")



st.set_page_config(page_title="AI Powerd TempCalc", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: black; /* Set background color to black */
    }
    .center-text {
        text-align: left; /* left the text */
        color: blue; /* Set font color to blue */
        font-size: 0.1em; /* Adjust font size */
        margin-top: 1.0px; /* Add space at the top */
    }
    .right-text{
         text-align: right; /* left the text */
        color: blue; /* Set font color to blue */
        font-size: 0.1em; /* Adjust font size */
        margin-top: 1.0px; /* Add space at the top */
    }
    
    </style>
    
    """,
    unsafe_allow_html=True
)


# Display title
st.markdown('<h1 class="title">AI- Powered TempCalc </h1>', unsafe_allow_html=True)
#st.write("Developed by Uzma Ilyas")
# Display centered text
st.markdown('<p class="left-text">An advanced calculator with Temperature conversion features with AI</p>', unsafe_allow_html=True)
user_input=st.text_input("Enter your Prompt")

#out put
st.markdown('<p class="right-text">Developed by Uzma Ilyas</p>',unsafe_allow_html=True)
if st.button("submit"):
    response=agent.invoke(user_input)
    st.write(response)