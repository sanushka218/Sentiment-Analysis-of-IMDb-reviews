"""
def linearsearch(lys, element):
    for i in range(len(lys)):
        if lys[i]== element:
            return i+1
    return -1
a=[1,2,4,6,3,88,23]
print(linearsearch(a,6))
"""
def binarysearch(lys,b,l, element):
    mid = int((b+l)/2)
    while b<l:
        if lys[mid] < element:
            binarysearch(lys,b,mid-1,element)
        else:
            if lys[mid] > element:
                binarysearch(lys,mid+1,l,element)
            else:
                return mid
a=[1,2,3,4,5,6,7,8,9,10]
print(binarysearch(a,0,len(a),6))



