#!/usr/bin/env python3
"""Module to calculate the integral of a polynomial."""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    Args:
        poly (list): A list of coefficients representing a polynomial.
        C (int): The integration constant.

    Returns:
        list: A new list of coefficients representing the integral,
              or None if poly or C are not valid.
    """
    # Validate the input types and constraints
    if type(poly) is not list or len(poly) == 0:
        return None
    if type(C) is not int and type(C) is not float:
        return None
    
    # Ensure C is an integer if it's a whole number float
    if type(C) is float and C % 1 == 0:
        C = int(C)

    # Check if all coefficients in the polynomial are numbers
    for coef in poly:
        if type(coef) is not int and type(coef) is not float:
            return None

    # Initialize the integral list with the constant C
    integral = [C]

    # Apply the reverse power rule to calculate the new coefficients
    for i in range(len(poly)):
        val = poly[i] / (i + 1)
        # Represent whole numbers as integers
        if val % 1 == 0:
            val = int(val)
        integral.append(val)

    # Remove trailing zeros to make the list as small as possible
    # We use len(integral) > 1 to ensure we don't pop a final zero if the list is just [0]
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
