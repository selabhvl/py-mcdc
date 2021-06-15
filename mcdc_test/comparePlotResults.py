h1=[1, 2, 3, 1, 2, 6, 3], [2, 4, 5, 8, 0, 5, 18], [1, 1, 1, 8, 0, 140, 3], [0, 1, 0, 8, 0, 10, 13]
h2=[1,2,3],[2,4,5],[1,1,1]
def compareresult(hs):
    BoolTF =list(map(lambda vals: list(map(lambda v: v == min(vals), vals)), (zip(*hs))))
    comparison = (list(map(lambda x: sum(x), zip(*BoolTF))))
    return comparison

Result = compareresult(h2)
print(Result)