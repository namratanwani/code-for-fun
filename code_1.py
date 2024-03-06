def square(n):
    return n**2


def fibonacci(n):
    """
    This function implements the Fibonacci sequence, where each number is the sum
    of the two preceding numbers, starting with 0 and 1. The function returns a
    list of all the numbers in the sequence up to the input `n`.

    Args:
        n (int): The `n` input parameter in the `fibonacci()` function determines
            the number of terms to compute and return in the Fibonacci sequence.
            It sets the limit for how many numbers the function will generate and
            return.

    Returns:
        list: The output of this function will be a list of numbers that follow
        the Fibonacci sequence, starting from 0 and 1, and then iteratively adding
        the previous two numbers in the list to get the next number.
        
        Therefore, the output of the `fibonacci(5)` function will be `[0, 1, 1,
        2, 3]`.

    """
    lst = [0,1]
    for i in range(n-2):
        s = lst[-1] + lst[-2]
        lst.append(s)
    return lst

result = square(4)
print(result)

fib_list = fibonacci(10)
print(fib_list)