stra = "我想吃炸鸡，你去不去"
strb = "我不想吃炸鸡，你去吧"

for i in range(len(stra)):
    print("a[{0}]={1}".format(i, stra[i]))
    b = i
    a = i + 1
    for j in range(i, len(strb)):
        if stra[i] == strb[j]:
            print('stra[{0}]-{1} == strb[{2}]-{3}'.format(i, stra[i], j, strb[j]))
            for k in range(1, len(strb)-j):
                if stra[i: i + k] == strb[j: j+k]:
                    print('stra[{0}:{1}]-{2} == strb[{3}:{4}]-{5}'.format(i, i+k, stra[i: i + k], j, j+k, strb[j: j+k]))
                else:
                    print(k)
        else:
            print('stra[{0}]-{1} != strb[{2}]-{3}'.format(i, stra[i], j, strb[j]))
    print()



    # if (stra[i] == strb[i]):
    #     print('{0}=={1}'.format(stra[i], strb[i]))
    # else:
    #     print('{0}!={1}'.format(stra[i], strb[i]))
    # for j in range(len(stra)):
    #     print(stra[i])
    #     print(strb[j])
