
'''
contract = [
    ['a','b','a','c'],
    ['c','d','a','e'],
    ['c','e','b','d'],
    #['d','e','a','b'],
]
'''
'''
contract = [
    ['a', 'c', 'b', 'e'],
    ['a', 'd', 'c', 'g'],
    ['b', 'f', 'd', 'h'],
    ['c', 'd', 'b', 'f'],
    ['d', 'e', 'a', 'i'],
    ['g', 'h', 'b', 'j'],
    ['g', 'h', 'g', 'j'],
    ['h', 'j', 'e', 'i'],
]
'''

'''
char_set = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n'])
contract = [
    ['e', 'f', 'k', 'd'],
    ['c', 'h', 'a', 'n'],
    ['j', 'n', 'j', 'l'],
    ['c', 'a', 'f', 'h'],
    ['j', 'l', 'e', 'n'],
    ['n', 'l', 'a', 'f'],
    ['d', 'i', 'k', 'n'],
    ['d', 'i', 'g', 'i'],
    ['c', 'l', 'g', 'k'],
    ['g', 'b', 'g', 'i'],
    ['g', 'i', 'd', 'm'],
    ['c', 'h', 'c', 'a'],
    ['e', 'f', 'h', 'l'],
    ['j', 'l', 'j', 'a'],
    ['k', 'm', 'e', 'i'],
    ['j', 'n', 'j', 'f'],
]
'''


char_set = set(['a', 'c', 'e', 'f', 'h', 'j', 'l', 'n'])
contract = [['c', 'h', 'a', 'n'], ['j', 'n', 'j', 'l'], ['c', 'a', 'f', 'h'], ['j', 'l', 'e', 'n'], ['n', 'l', 'a', 'f'], ['c', 'h', 'c', 'a'], ['e', 'f', 'h', 'l'], ['j', 'l', 'j', 'a'], ['j', 'n', 'j', 'f']]

'''
char_set = set(['b', 'd', 'g', 'i'])
contract = [['d', 'i', 'g', 'i'], ['g', 'b', 'g', 'i']]
'''
#leaf_max = 14


S = [None] * len(char_set)
L = [None] * len(char_set)
set_id = {}

Q = []

for i, l in enumerate(char_set):
#for i in range(leaf_max):
    #l = chr(97+i)
    S[i] = set(l)
    L[i] = []
    set_id[l] = i

for c in contract:
    imp = (c[2], c[3], c[0], c[3])
    print("Contract: %s   Imp: %s" %(c, imp))
    L[set_id[c[2]]].append(imp)
    L[set_id[c[3]]].append(imp)
    Q.append((c[0], c[1]))

#print("L:", L)

print("Queue: ", Q)
while len(Q) > 0:
    p, q = Q.pop(0)
    Sp = S[set_id[p]]
    Sq = S[set_id[q]]

    if set_id[p] != set_id[q]:
        print()
        print("Looking at sets %d %d  cmd: (%s, %s)"%(set_id[p], set_id[q], p,q))
        print("%s: %s   %s: %s"%(p,S[set_id[p]],q,S[set_id[q]]))
        if len(L[set_id[p]]) < len(L[set_id[q]]):
            l = L[set_id[p]]
        else:
            l = L[set_id[q]]

        print("l: ", l)
        for imp in l:
            print("Imp:", imp)
            u,v,x,y = imp
            if (u in Sp and v in Sq) or (v in Sp and u in Sq):
                print("Adding cmd (%s,%s)"%(x,y))
                Q.append((x,y))

        
        print("Bye set: %s (%d)"%(q,set_id[q]))

        L[set_id[p]] += L[set_id[q]]
        L[set_id[q]] = None
        print("New l: ", L[set_id[p]])

        print("Before update")
        print("Sp: ", S[set_id[p]], Sp)
        print("Sq: ", S[set_id[q]], Sq)
        S[set_id[p]].update(S[set_id[q]])
        S[set_id[q]] = None
        print("After update")
        print("Sp: ",S[set_id[p]])
        print("Sq: ",S[set_id[q]])

        # Make sure all leafs have correct id
        print("before set_id: ", set_id)
        for x in S[set_id[p]]:
            set_id[x] = set_id[p]
        print("after set_id: ", set_id)


def renormalise_contract(leaf_set, contract):
    c_map = {char:chr(97+i) for i, char in enumerate(leaf_set)}
    #print("Mapping: ",c_map)
    new_contract = [None]*len(contract)
    for i in range(len(contract)):
        new_contract[i] = [None]*4
        for j in range(4):
            new_contract[i][j] = c_map[contract[i][j]]

    return new_contract


print()
print("Final sets")
for i in range(len(char_set)):
    if S[i]!=None:
        print("Subset: ",sorted(list(S[i])))

        new_rules = []
        for c in contract:
            if S[i].issuperset(set(c)):
                new_rules.append(c)

        #print("L: ",L[i])
        #print("R: ",new_rules)
        #n = renormalise_contract(S[i], new_rules)
        print("Constraints: ", new_rules)