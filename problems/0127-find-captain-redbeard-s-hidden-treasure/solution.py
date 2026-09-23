def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

  Returns:
        float: The x-coordinate of the minimum point.
    """
    alpha = 0.08
    x = start_x
    n = 100
    for i in range(n):
      der = 4*(x**3) - 9*(x**2)
      x -= alpha*der
    return x