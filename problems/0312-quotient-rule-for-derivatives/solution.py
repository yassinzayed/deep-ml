import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    def evaluate_funcs(func_list, x):
        final_values = []
        for i in func_list:
            sum = 0
            reversed_i = i[::-1]
            for k in range(len(i)):
                sum += reversed_i[k]*(x**k)
            final_values.append(sum)
        return final_values

    def derivative_calc(func_list):
        derivative_list = []
        for i in func_list:
            derivative_coeffs = [0] * len(i)
            reversed_i = i[::-1]
            for k in range(len(i)):
                if k < (len(i)-1):
                    derivative_coeffs[k] = reversed_i[k+1]*(k+1)
                else:
                    derivative_coeffs[k] = 0
            derivative_coeffs.reverse()
            derivative_list.append(derivative_coeffs)
        return derivative_list
    
    def quotient_rule(func_list, x):
        function_values = evaluate_funcs(func_list, x)
        derivative_funcs = derivative_calc(func_list)
        derivative_values = evaluate_funcs(derivative_funcs, x)
        denom = function_values[1]**2
        nom = function_values[1]*derivative_values[0] - derivative_values[1]*function_values[0]
        return nom/denom
    
    func_lists = [g_coeffs, h_coeffs]
    return quotient_rule(func_lists, x)

