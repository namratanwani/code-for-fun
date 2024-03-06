def square(n):
    return n**2

def square2(n):
    ans = n*n
    return ans

def fibonacci(n):
    lst = [0,1]
    for i in range(n-2):
        s = lst[-1] + lst[-2]
        lst.append(s)
    return lst

def gcd(a, b):

	if (a == 0):
		return b
	if (b == 0):
		return a

	if (a == b):
		return a
	if (a > b):
		return gcd(a-b, b)
	return gcd(a, b-a)


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