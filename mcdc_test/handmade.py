from pyeda.inter import *
a, b, c, d, e, f, g, h, i, j, k = map(bddvar, 'abcdefghij')
D1 = a & b & c | d & e & f | g & h & a
D2 = a & b & c | c & d & e | e & f & g
D3 = a & b | b & c | c & d | d & a
D4 = a & f & ~c & d & ~e | ~a & f & h & i & j | b & ~g & ~c & ~d & ~e | ~b & g & ~h & ~i & ~j
D5 = b & ~g & ~c & ~d & ~e | ~a & c & d & e

hdmd = [D1, D2, D3, D4, D5]