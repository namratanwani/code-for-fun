def square(n):
    return n**2


def fibonacci(n):
    lst = [0,1]
    for i in range(n-2):
        s = lst[-1] + lst[-2]
        lst.append(s)
    return lst

result = square(4)
print(result)

fib_list = fibonacci(10)
print(fib_list)