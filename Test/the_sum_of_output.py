import numpy as np

if __name__ == "__main__":
    number_of_successes = 0
    total_number = 10000000
    index = 0
    while index < total_number:
        index += 1
        ran_numbers = np.random.random(size=(10,))
        if sum(ran_numbers) < 1:
            number_of_successes += 1
    print(number_of_successes / total_number)