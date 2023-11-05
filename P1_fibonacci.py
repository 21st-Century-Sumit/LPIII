def fibonacci_iterative(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


print(fibonacci_iterative(5))
# Non-Recursive (Iterative) Approach: Time Complexity: O(n)


def fibonacci_recursive(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


print(fibonacci_recursive(5))
# Recursive Approach: Time Complexity: O(2^n)
