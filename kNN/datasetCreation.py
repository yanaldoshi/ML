import random

def main(filename):
    fref=open(filename,'w')
    for i in xrange(1000):
        temp=[1,2,3]
        temp[0]=random.randrange(100,10000000)
        temp[1]=random.randrange(100,temp[0])
        ratio=temp[0]/float(temp[1])
        if(ratio<1.7):
            temp[2]=2
        else:
            temp[2]=1
        if(random.random()>0.9):
            if(temp[2]==1): temp[2]=2
            else: temp[2]=1
        temp_str=str(temp[0])+'\t'+str(temp[1])+'\t'+str(temp[2])+'\n'
        fref.write(temp_str)
    fref.close()

if __name__=='__main__':
    main('Population2Disease.txt')
