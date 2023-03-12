def batch(l: list, n: int) -> list:
    for i in range(0, len(l), n):
        yield l[i:i + n]
