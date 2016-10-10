while True:
    datain = input('num or char\n')
    if type(datain) == int:
        templist=[]
        tempint=datain
        while tempint/26 != 0:
            templist.append(chr(64+tempint%26))
            tempint=tempint/27
        templist.append(chr(64 + tempint))
        for i in range(len(templist)):
            print templist[-1-i],
        print ''
    else:
        sum=0
        for i in range(len(datain)):
            sum=sum+(ord(datain[-1-i])-64)*(26**i)
        print sum+1
