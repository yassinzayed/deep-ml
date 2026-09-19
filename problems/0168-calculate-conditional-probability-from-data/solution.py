def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    nom = 0
    denom = 0
    for i in data:
      if i[0] == x and i[1] != y:
        denom += 1
      elif i[0] == x and i[1] == y:
        denom += 1
        nom += 1
    if denom == 0:
      denom = 1
    return nom/denom