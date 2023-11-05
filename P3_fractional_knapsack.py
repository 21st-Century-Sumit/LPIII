def fractional_knapsack(items, capacity):
    # Calculate the value-to-weight ratio for each item
    for item in items:
        item['ratio'] = item['value'] / item['weight']

    # Sort the items based on their value-to-weight ratios in descending order
    items.sort(key=lambda x: x['ratio'], reverse=True)

    total_value = 0.0
    knapsack = []

    for item in items:
        if item['weight'] <= capacity:
            knapsack.append(item)
            total_value += item['value']
            capacity -= item['weight']
        else:
            fraction = capacity / item['weight']
            knapsack.append({'weight': capacity, 'value': fraction * item['value']})
            total_value += fraction * item['value']
            break

    return total_value, knapsack


# Example usage
items = [
    {'weight': 10, 'value': 60},
    {'weight': 20, 'value': 100},
    {'weight': 30, 'value': 120},
]

knapsack_capacity = 50
max_value, selected_items = fractional_knapsack(items, knapsack_capacity)

print(f"Maximum value in the knapsack: {max_value}")
print("Selected items:")
for item in selected_items:
    print(f"Weight: {item['weight']}, Value: {item['value']}")
