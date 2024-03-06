def square(n):
    return n**2

def square2(n):
    """
    This Python function takes a single argument `n` and returns the value of
    `n^2`. In other words, it multiplies the input `n` by itself to get the output.

    Args:
        n (int): In this function, `n` is a formal parameter that represents the
            number to be squared. The function takes an integer `n` as input and
            returns the result of square it (i.e., `n*n`). So, `n` does the job
            of providing the value that gets squared.

    Returns:
        int: The output returned by the `square2` function is the square of the
        input `n`. In other words, if `n` is 3, then the output will be `3*3 = 9`.

    """
    ans = n*n
    return ans

def fibonacci(n):
    """
    This function implements the Fibonacci sequence algorithm to generate a list
    of numbers from 0 to `n`. Here's what it does:
    1/ It creates a list called `lst` with two initial elements `[0, 1]`.
    2/ It loops from `n-2` to 0 using `range`, and for each iteration, it appends
    the sum of the last two elements in the list to the current list.
    3/ At each iteration, the last two elements are `lst[-1]` and `lst[-2]`.
    4/ Finally, it returns the generated list `lst`.

    Args:
        n (int): The `n` input parameter in the given code defines the length of
            the Fibonacci sequence that the function should generate. In other
            words, it specifies how many terms the function should calculate and
            return as a list.
            
            In essence, the value of `n` determines the output of the function.
            If `n` is set to a small value, like 2 or 3, the function will only
            generate a few terms of the Fibonacci sequence, while larger values
            of `n` will result in a longer list of terms being returned.

    Returns:
        list: The output of this function is a list of numbers in Fibonacci sequence,
        where each number is the sum of the previous two numbers in the list. In
        other words, the output will be: `[0, 1, 1, 2, 3, 5, 8, 13, ...]`

    """
    lst = [0,1]
    for i in range(n-2):
        s = lst[-1] + lst[-2]
        lst.append(s)
    return lst

def gcd(a, b):

	"""
	This function calculates the greatest common divisor (GCD) of two integers `a`
	and `b`. It works by using a recursive approach, checking various conditions to
	determine the GCD. The function returns the GCD if found, or `b` if `a` is equal
	to zero, or `a` if `b` is equal to zero, or `a` if `a > b`, or `b` if `a - b` is
	smaller than `b`, or the GCD of `a` and `b - a`.

	Args:
	    a (int): The `a` input parameter in the given function `gcd()` serves as a
	        "seed" value for the Greatest Common Divisor (GCD) calculation. When `a`
	        is not equal to `0`, it is used as the initial value for the GCD calculation.
	        When `a` is equal to `0`, the function returns `b`. In either case, the
	        remaining portion of the function code ensures that the GCD of `a` and
	        `b` is found by recursively calling itself with the appropriate parameters.
	    b (int): The `b` input parameter in the given implementation of the greatest
	        common divisor (GCD) algorithm serves as a counterpart to `a`. It is used
	        to determine the GCD of two numbers `a` and `b`. Specifically, it helps
	        to recursively calculate the GCD by providing an alternative base value
	        for the recursion when `a` and `b` have no common factors.

	Returns:
	    int: The function `gcd` returns the greatest common divisor of two integers
	    `a` and `b`. The function works by recursively calling itself until it finds
	    the GCD.
	    
	    The function will always return a non-zero value, since it checks if the input
	    numbers are zero or equal to each other before making any recursive calls.
	    If both inputs are zero, the function returns the other input. If one input
	    is greater than the other, the function recursively calls itself with the
	    smaller input subtracted from the larger input.
	    
	    Therefore, the output of the function `gcd(a, b)` will always be a non-zero
	    integer, regardless of the values of `a` and `b`.

	"""
	if (a == 0):
		return b
	if (b == 0):
		return a

	if (a == b):
		return a
	if (a > b):
		return gcd(a-b, b)
	return gcd(a, b-a)



class User:
    """
        A class named User.
    """
    def __init__(self, name, age):
        """
        self.name = name creates an attribute called name and assigns the value of the name parameter to it.
        self.age = age creates an attribute called age and assigns the value of the age parameter to it.
        """
        self.name = name
        self.age = age

    # Instance method
    def description(self):
        return f"{self.name} is {self.age} years old"

    # Another instance method
    def speak(self, sound):
        return f"{self.name} says {sound}"


a = 98
b = 56
if(gcd(a, b)):
	print('GCD:', gcd(a, b))
else:
	print('not found')


result = square(4)
print(result)

result = square2(5)
print(result)

fib_list = fibonacci(10)
print(fib_list)