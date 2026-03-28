#!/usr/bin/env python3
"""Module to calculate the derivative of a polynomial."""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    Args:
        poly (list): A list of coefficients representing a polynomial.
                     The index of the list represents the power of x.

    Returns:
        list: A new list of coefficients representing the derivative,
              or None if poly is not valid.
    """
    if type(poly) is not list or len(poly) == 0:
        return None

    # Verify all elements are valid numbers (integers or floats)
    for coef in poly:
        if type(coef) is not int and type(coef) is not float:
            return None

    # If the polynomial is just a constant (e.g., [5]), derivative is 0
    if len(poly) == 1:
        return [0]

    # Calculate the derivative: coefficient * power (which is the index i)
    derivative = [poly[i] * i for i in range(1, len(poly))]

    # Handle the edge case where the resulting derivative is entirely zeros
    if all(coef == 0 for coef in derivative):
        return [0]

    return derivative
