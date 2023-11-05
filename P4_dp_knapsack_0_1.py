def knapsack_dynamic_programming(values, weights, capacity):
    n = len(values)

    # Create a 2D table to store the maximum value for each subproblem
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Build the table in a bottom-up manner
    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
            else:
                dp[i][w] = dp[i - 1][w]

    # Backtrack to find the selected items
    selected_items = []
    i, j = n, capacity
    while i > 0 and j > 0:
        if dp[i][j] != dp[i - 1][j]:
            selected_items.append(i - 1)
            j -= weights[i - 1]
        i -= 1

    selected_items.reverse()

    return dp[n][capacity], selected_items


# Example usage
values = [60, 100, 120]
weights = [10, 20, 30]
knapsack_capacity = 50
max_value, selected_items = knapsack_dynamic_programming(values, weights, knapsack_capacity)

print(f"Maximum value in the knapsack: {max_value}")
print("Selected items:")
for item in selected_items:
    print(f"Weight: {weights[item]}, Value: {values[item]}")
