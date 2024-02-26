import numpy as np
import pandas as pd

func_list = [
    'getitem', 'select',
    'numpy.array', 'numpy.ndarray', 'numpy.argmax', 'numpy.argmin', 'numpy.max', 'numpy.min', 'numpy.sum',
    'numpy.add', 'numpy.subtract', 'numpy.multiply', 'numpy.divide', 'numpy.<', 'numpy.<=', 'numpy.>', 'numpy.>=',
    'numpy.==', 'numpy.!=', 'numpy.mean', 'numpy.median', 'numpy.std', 'numpy.var', 'numpy.abs', 'numpy.sqrt',
    'numpy.square', 'numpy.log', 'numpy.exp', 'numpy.power', 'numpy.sort', 'numpy.delete', 'numpy.all', 'numpy.any',
    'numpy.diff', 'numpy.corrcoef', 'numpy.cov',
    'np.array', 'np.ndarray', 'np.argmax', 'np.argmin', 'np.max', 'np.min', 'np.sum',
    'np.add', 'np.subtract', 'np.multiply', 'np.divide', 'np.<', 'np.<=', 'np.>', 'np.>=',
    'np.==', 'np.!=', 'np.mean', 'np.median', 'np.std', 'np.var', 'np.abs', 'np.sqrt',
    'np.square', 'np.log', 'np.exp', 'np.power', 'np.sort', 'np.delete', 'np.all', 'np.any',
    'np.diff', 'np.corrcoef', 'np.cov',
    'max', 'min', 'sum', 'len', 'str', 'int', 'float', 'abs', 'round', '<', '<=', '>', '>=', '==', '!='
]


def getitem(a, b):
    if type(a) is list or type(a) is np.ndarray:
        return a[int(b)]
    elif type(a) is list or type(a) is np.ndarray:
        return a[int(b)]
    raise TypeError("type error")


def exec_np(func: str, args: list):
    if func == 'array':
        return np.array(args)
    elif func == 'ndarray':
        return np.ndarray(args)
    elif func == 'argmax':
        return np.argmax(args)
    elif func == 'argmin':
        return np.argmin(args)
    elif func == 'max':
        return np.max(args)
    elif func == 'min':
        return np.min(args)
    elif func == 'sum':
        return np.sum(args)
    elif func == 'add':
        if len(args) != 2:
            raise ValueError("args length error")
        return np.add(args[0], args[1])
    elif func == 'subtract':
        if len(args) != 2:
            raise ValueError("args length error")
        return np.subtract(args[0], args[1])
    elif func == 'multiply':
        if len(args) != 2:
            raise ValueError("args length error")
        return np.multiply(args[0], args[1])
    elif func == 'divide':
        if len(args) != 2:
            raise ValueError("args length error")
        if args[1] == 0:
            raise ValueError("divide by zero")
        return np.divide(args[0], args[1])
    elif func == '<':
        if len(args) != 2:
            raise ValueError("args length error")
        return np.array(args[0]) < args[1]
    elif func == '<=':
        if len(args) != 2:
            raise ValueError("args length error")
        return np.array(args[0]) <= args[1]
    elif func == '>':
        if len(args) != 2:
            raise ValueError("args length error")
        return np.array(args[0]) > args[1]
    elif func == '>=':
        if len(args) != 2:
            raise ValueError("args length error")
        return np.array(args[0]) >= args[1]
    elif func == '==':
        if len(args) != 2:
            raise ValueError("args length error")
        return np.array(args[0]) == args[1]
    elif func == '!=':
        if len(args) != 2:
            raise ValueError("args length error")
        return np.array(args[0]) != args[1]
    elif func == 'mean':
        return np.mean(args)
    elif func == 'median':
        return np.median(args)
    elif func == 'std':
        return np.std(args)
    elif func == 'var':
        return np.var(args)
    elif func == 'abs':
        return np.abs(args)
    elif func == 'sqrt':
        return np.sqrt(args)
    elif func == 'square':
        return np.square(args)
    elif func == 'log':
        return np.log(args)
    elif func == 'exp':
        return np.exp(args)
    elif func == 'power':
        return np.power(args)
    elif func == 'sort':
        return np.sort(args)
    elif func == 'delete':
        if len(args) != 2:
            raise ValueError("args length error")
        return np.delete(args[0], int(args[1]))
    elif func == 'all':
        return bool(np.all(args))
    elif func == 'any':
        return bool(np.any(args))
    elif func == 'diff':
        res = np.diff(args)
        if len(res) == 1:
            return res[0]
        else:
            return res
    elif func == 'corrcoef':
        if np.std(args[0]) == 0 or np.std(args[1]) == 0:
            raise ValueError("std is zero")
        return np.corrcoef(args[0], args[1])[0, 1]
    elif func == 'cov':
        return np.cov(args[0], args[1])[0, 1]
    else:
        raise ValueError("numpy func name error: {}".format(func))


def exec_normal(func: str, args: list):
    if func == 'max':
        if len(args) != 2:
            raise ValueError("args length error")
        return max(args[0], args[1])
    elif func == 'min':
        if len(args) != 2:
            raise ValueError("args length error")
        return min(args[0], args[1])
    elif func == 'sum':
        return sum(args)
    elif func == 'len':
        return len(args)
    elif func == 'str':
        if len(args) != 1:
            raise ValueError("args length error")
        return str(args[0])
    elif func == 'int':
        if len(args) != 1:
            raise ValueError("args length error")
        return int(args[0])
    elif func == 'float':
        if len(args) != 1:
            raise ValueError("args length error")
        return float(args[0])
    elif func == 'abs':
        if len(args) != 1:
            raise ValueError("args length error")
        return abs(args[0])
    elif func == 'round':
        if len(args) != 1:
            raise ValueError("args length error")
        return round(args[0])
    elif func == 'getitem':
        if len(args) != 2:
            raise ValueError("args length error")
        return getitem(args[0], args[1])
    elif func == '<':
        if len(args) != 2:
            raise ValueError("args length error")
        return bool(args[0] < args[1])
    elif func == '<=':
        if len(args) != 2:
            raise ValueError("args length error")
        return bool(args[0] <= args[1])
    elif func == '>':
        if len(args) != 2:
            raise ValueError("args length error")
        return bool(args[0] > args[1])
    elif func == '>=':
        if len(args) != 2:
            raise ValueError("args length error")
        return bool(args[0] >= args[1])
    elif func == '==':
        if len(args) != 2:
            raise ValueError("args length error")
        return bool(args[0] == args[1])
    elif func == '!=':
        if len(args) != 2:
            raise ValueError("args length error")
        return bool(args[0] != args[1])
    else:
        raise ValueError("normal func name error: {}".format(func))
