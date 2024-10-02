def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y == 0:
        return "Error: Division by zero!"
    return x / y

def calculate(operation, x, y):
    if operation == '+':
        return add(x, y)
    elif operation == '-':
        return subtract(x, y)
    elif operation == '*':
        return multiply(x, y)
    elif operation == '/':
        return divide(x, y)
    else:
        return "Error: Invalid operation!"

    def main():
        operation, x, y = get_user_input()
        result = calculate(operation, x, y)
        print(f"The result is: {result}")

    if __name__ == "__main__":
        main()