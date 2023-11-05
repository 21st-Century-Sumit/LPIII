import random


def quicksort_deterministic(arr):
    if len(arr) <= 1:
        return arr, 0

    comparisons = 0

    # Choose the last element as the pivot
    pivot = arr[-1]
    left = []
    right = []

    for element in arr[:-1]:
        comparisons += 1
        if element <= pivot:
            left.append(element)
        else:
            right.append(element)

    sorted_left, comparisons_left = quicksort_deterministic(left)
    sorted_right, comparisons_right = quicksort_deterministic(right)

    return sorted_left + [pivot] + sorted_right, comparisons + comparisons_left + comparisons_right


def quicksort_randomized(arr):
    if len(arr) <= 1:
        return arr, 0

    comparisons = 0

    # Choose a random pivot
    pivot_index = random.randint(0, len(arr) - 1)
    pivot = arr[pivot_index]
    left = []
    right = []

    for i, element in enumerate(arr):
        if i != pivot_index:
            comparisons += 1
            if element <= pivot:
                left.append(element)
            else:
                right.append(element)

    sorted_left, comparisons_left = quicksort_randomized(left)
    sorted_right, comparisons_right = quicksort_randomized(right)

    return sorted_left + [pivot] + sorted_right, comparisons + comparisons_left + comparisons_right


# Test and compare the two variants
arr = [3, 6, 8, 10, 1, 2, 1]

print("Original array:", arr)

sorted_deterministic, comparisons_deterministic = quicksort_deterministic(arr.copy())
print("Deterministic Quick Sort:")
print("Sorted array:", sorted_deterministic)
print("Comparisons made:", comparisons_deterministic)

sorted_randomized, comparisons_randomized = quicksort_randomized(arr.copy())
print("\nRandomized Quick Sort:")
print("Sorted array:", sorted_randomized)
print("Comparisons made:", comparisons_randomized)
