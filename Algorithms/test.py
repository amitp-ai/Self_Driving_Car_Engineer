'''
import sys
#print(sys.version_info[0:]) #prints the python verison

import numpy as np

# n1 = np.array([[1,2,3],[11,22,33]])
# print(n1)
# print(n1.shape)

# for i in n1:
# 	print(i)


# a = [1,2,3,4,5]
# b = [11,22,33,44]
# print('zip')
# for i,j in zip(a,b):
# 	print(i,j)

# print('enumerate')
# for i,j in enumerate(b):
# 	print(i,j)

# print()
# a = np.random.rand(2,2)
# b = np.random.rand(2,2)
# c = np.random.rand(2,2)

# lst = [a,b,c]
# print(lst)

# np_lst = np.array(lst)
# print()
# print(np_lst.shape)
# print(np_lst)


"""
Multiline python comments
continue with line 2 etc
"""

import os
def inference(file_name):
	print(file_name + ' newstring')
	print(file_name.replace(os.path.basename(file_name), ''))


if __name__ == '__main__':
	infile = sys.argv[-1]
	inference(infile)

a = 'amitpatel'
print(a[0:2])

var1 = None
print(var1)
'''

var1 = [1,2]
var2 = var1
var2.append(100) #this keeps the alias
print(var1)
var2 = [1,2,3,4,5] #breaks the alias
print(var1)

var1 = int(10) #using constructor form
print(var1)
var1 = 10 #using literal form (onlys works for builtin datatypes). Gives same result as above.

var1 = [1,2,3] #literal form
print(var1)
var2 = list([1,2,3]) #using constructor
print(var2)

var1 = bool()
print(var1)

var1 = 1
print(type(var1))
var1 = float(1)
print(type(var1))
var1 = 1.0
print(type(var1))

str1 = 'C:drive\\amit'
print(str1)

set1 = {}
print(type(set1))
set1 = set()
print(type(set1))

var1 = 1.0
var2 = var1
print(var1 is var2)
var2 = 1.0
print(var1 is var2)
print(var1 == var2)

# print('Enter a number')
# var1 = input() #sublime text doesn't support console input
# print(var1)

# Keyword argument #
def my_function(a=1,b=2,c=3):
	print(a,b,c)
	return None

my_function()
my_function(c=111,a=1212)
# Keyword argument #

# import pdb; pdb.set_trace() #python debugger


#Python Generators#
def factors(n): # generator that computes factors
	""" This is an example of a generator """
	for k in range(1,n+1):
		if n % k == 0: # divides evenly, thus k is a factor
			yield k
			yield 0 #can have as many yields as we want to
	print(dir())

print(factors(10))
print(factors(10))\

for factor in factors(3):
	print(factor)
#Python Generators#

a,b,c,d = factors(3) #can do this for any iterable and generators are iterables
print(a,b,c,d)

# print()
# print(dir()) #lists all the identifiers i.e. object and function names (identifiers are like pointers in C++) in the current namespace
# print(vars())

print(help(factors)) #prints the docstring
