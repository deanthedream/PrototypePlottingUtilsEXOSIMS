import os


lb=60
ub=62
date = '29May18'
assert lb < ub, "lb is not less than ub"
for i in range(lb,ub+1):
    myString = os.getcwd() + '/Dean' + date + 'RS' + '09' + 'CXXfZ01OB' + "%02d"%i + 'PP03SU03'
    try:
        os.mkdir(myString)
        print 'MADE DIR: ' + myString
    except:
        print 'DID NOT MAKE DIR: ' + myString
print "Done Making Dirs"

for i in range(lb,ub+1):
    myStringLOCAL = os.getcwd() + '/Dean' + date + 'RS' + '09' + 'CXXfZ01OB' + "%02d"%i + 'PP03SU03/.'
    myStringATUIN = '/data2/extmount/EXOSIMSres' + '/Dean' + date + 'RS' + '09' + 'CXXfZ01OB' + "%02d"%i + 'PP03SU03/*'
    command = 'scp ' + 'drk94@atuin.coecis.cornell.edu:' + myStringATUIN + ' ' + myStringLOCAL + ' >> /dev/null'
    print command
    os.system(command)

print 'Done moving files'


# Starting Point for creating a createRunDirectories that uses the queue file to download runs.
import os


jsonNames = ["Dean6June18RS09CXXfZ01OB56PP01SU01.json","Dean6June18RS09CXXfZ01OB01PP01SU01.json"]
dirNames = [jsonNames[i].split('.')[0] for i in range(len(jsonNames))]


for i in range(len(dirNames)):
    myString = '/home/dean/Documents/SIOSlab/' + dirNames[i]
    try:
        os.mkdir(myString)
        print 'MADE DIR: ' + myString
    except:
        print 'DID NOT MAKE DIR: ' + myString
print "Done Making Dirs"

for i in range(len(dirNames)):
    myStringLOCAL = '/home/dean/Documents/SIOSlab/' + dirNames[i] + '/.'
    myStringATUIN = '/data2/extmount/EXOSIMSres/' + dirNames[i] + '/*'
    command = 'scp ' + 'drk94@atuin.coecis.cornell.edu:' + myStringATUIN + ' ' + myStringLOCAL + ' >> /dev/null'
    print command
    os.system(command)

