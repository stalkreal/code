# --- Recursive Fibonacci with Step Count ---
def fibonacci_recursive(n, step_count):
    step_count += 1  # count every function call

    if n <= 1:
        return n, step_count

    f1, step_count = fibonacci_recursive(n - 1, step_count)
    f2, step_count = fibonacci_recursive(n - 2, step_count)
    return f1 + f2, step_count


def fibonacci_recursive_series(n):
    series = []
    total_steps = 0
    for i in range(n):
        value, total_steps = fibonacci_recursive(i, total_steps)
        series.append(value)
    return series, total_steps


# --- Iterative Fibonacci with Step Count ---
def fibonacci_iterative(n):
    steps = 0
    series = []
    a, b = 0, 1

    if n >= 1:
        series.append(a)
    if n >= 2:
        series.append(b)

    for _ in range(2, n):
        steps += 1
        c = a + b
        series.append(c)
        a, b = b, c

    return series, steps


# --- Main Program ---
n = int(input("Enter the number of terms: "))

# Iterative Output
iter_series, iter_steps = fibonacci_iterative(n)
print("\n--- Iterative Fibonacci ---")
print("Series:", *iter_series)
print("Step Count:", iter_steps)

# Recursive Output
rec_series, rec_steps = fibonacci_recursive_series(n)
print("\n--- Recursive Fibonacci ---")
print("Series:", *rec_series)
print("Step Count:", rec_steps)
