"""
Basic utility tools for agents.

These are example tools that can be used with any agent.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "sqrt(16)")
        
    Returns:
        Result of the calculation as a string
    """
    try:
        # Safe evaluation with limited builtins
        allowed_names = {
            "abs": abs,
            "max": max,
            "min": min,
            "pow": pow,
            "round": round,
            "sum": sum,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "ceil": math.ceil,
            "floor": math.floor,
            "pi": math.pi,
            "e": math.e,
        }
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get the current date and time.
    
    Args:
        format: Date/time format string (default: "2024-01-15 14:30:00")
        
    Returns:
        Current date and time as a formatted string
    """
    return datetime.now().strftime(format)


@tool
def text_processor(text: str, operation: str) -> str:
    """
    Process text with various operations.
    
    Args:
        text: The text to process
        operation: The operation to perform (uppercase, lowercase, reverse, word_count, char_count, lines)
        
    Returns:
        Processed text result
    """
    operation = operation.lower()
    
    if operation == "uppercase":
        return text.upper()
    elif operation == "lowercase":
        return text.lower()
    elif operation == "reverse":
        return text[::-1]
    elif operation == "word_count":
        return str(len(text.split()))
    elif operation == "char_count":
        return str(len(text))
    elif operation == "lines":
        return str(len(text.splitlines()))
    else:
        return f"Unknown operation: {operation}. Available: uppercase, lowercase, reverse, word_count, char_count, lines"


@tool
def json_formatter(data: str, indent: int = 2) -> str:
    """
    Format and validate JSON data.
    
    Args:
        data: JSON string to format
        indent: Number of spaces for indentation
        
    Returns:
        Formatted JSON string
    """
    try:
        parsed = json.loads(data)
        return json.dumps(parsed, indent=indent, ensure_ascii=False)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {str(e)}"


@tool
def data_converter(value: str, from_unit: str, to_unit: str) -> str:
    """
    Convert between common units.
    
    Args:
        value: The value to convert
        from_unit: Source unit (e.g., km, miles, celsius, fahrenheit, kg, lbs)
        to_unit: Target unit
        
    Returns:
        Converted value as a string
    """
    try:
        val = float(value)
    except ValueError:
        return f"Invalid number: {value}"
    
    # Length conversions
    length_units = {
        "km": 1000, "m": 1, "cm": 0.01, "mm": 0.001,
        "miles": 1609.34, "yards": 0.9144, "feet": 0.3048, "inches": 0.0254
    }
    
    # Weight conversions
    weight_units = {
        "kg": 1, "g": 0.001, "mg": 0.000001,
        "lbs": 0.453592, "oz": 0.0283495
    }
    
    # Temperature conversions
    def celsius_to_fahrenheit(c): return (c * 9/5) + 32
    def celsius_to_kelvin(c): return c + 273.15
    def fahrenheit_to_celsius(f): return (f - 32) * 5/9
    def kelvin_to_celsius(k): return k - 273.15
    
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    # Length conversion
    if from_unit in length_units and to_unit in length_units:
        meters = val * length_units[from_unit]
        result = meters / length_units[to_unit]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    
    # Weight conversion
    if from_unit in weight_units and to_unit in weight_units:
        kg = val * weight_units[from_unit]
        result = kg / weight_units[to_unit]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    
    # Temperature conversion
    if from_unit == "celsius" and to_unit == "fahrenheit":
        return f"{value}°C = {celsius_to_fahrenheit(val):.2f}°F"
    elif from_unit == "celsius" and to_unit == "kelvin":
        return f"{value}°C = {celsius_to_kelvin(val):.2f}K"
    elif from_unit == "fahrenheit" and to_unit == "celsius":
        return f"{value}°F = {fahrenheit_to_celsius(val):.2f}°C"
    elif from_unit == "fahrenheit" and to_unit == "kelvin":
        return f"{value}°F = {celsius_to_kelvin(fahrenheit_to_celsius(val)):.2f}K"
    elif from_unit == "kelvin" and to_unit == "celsius":
        return f"{value}K = {kelvin_to_celsius(val):.2f}°C"
    elif from_unit == "kelvin" and to_unit == "fahrenheit":
        return f"{value}K = {celsius_to_fahrenheit(kelvin_to_celsius(val)):.2f}°F"
    
    return f"Conversion from {from_unit} to {to_unit} not supported"


# Tool aliases for easier access
CalculatorTool = calculator
CurrentTimeTool = current_time
TextProcessorTool = text_processor
JSONFormatterTool = json_formatter
DataConverterTool = data_converter
