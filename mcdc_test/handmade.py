from pyeda.inter import *
a, b, c, d, e, f, g, h, i, j= map(bddvar, 'abcdefghij')
D21 = a & b & c | d & e & f | g & h & a
D22 = a & b & c | c & d & e | e & f & g
D23 = a & b | b & c | c & d | d & a
D24 = a & ~f & c & d & e | ~a & f & h & i & j | b & ~g & ~c & ~d & ~e | ~b & g & ~h & ~i & ~j
D25 = ~b & ~c & ~d & ~e | ~a & c & d & e

hdmd = [D21, D22, D23, D24, D25]
hdmd_names = ["D2"+str(i) for i in range(1, 6)]
#print(hdmd_names)
hdmd_num_cond = [8, 7, 4, 10, 5]
#hdmd_dict_name = {ds_i: value for ds_i, value in zip(hdmd_names,  hdmd_num_cond)}
hdmd_dict = dict(zip(hdmd, hdmd_num_cond))