import random

def random_result(context, event):
    return {"result" : str(random.random())}
