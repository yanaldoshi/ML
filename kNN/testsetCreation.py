import random

def main(filename):
    fref=open(filename,'w')
    for i in xrange(100):
        temp=[1,2,3]
        temp[0]=random.randrange(100,10000000)
        temp[1]=random.randrange(100,temp[0])
        temp_str=str(temp[0])+'\t'+str(temp[1])+'\t'+'\n'
        fref.write(temp_str)
    fref.close()

if __name__=='__main__':
    main('Population2DiseaseTesting.txt')
